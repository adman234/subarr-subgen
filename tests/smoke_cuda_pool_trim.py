"""GPU smoke for patch 0058 (#77): a failed load's CUDA pool memory is released.

Run INSIDE the built image, WITH a GPU:

    docker run --rm --gpus all -v "$PWD/upstream:/work:ro" -v "$PWD/tests:/smoke:ro" \
      -w /work -e PYTHONPATH=/work -e TRANSCRIBE_DEVICE=cuda --entrypoint python3 \
      ghcr.io/coaxk/subarr-subgen:<tag> /smoke/smoke_cuda_pool_trim.py

WHY THIS EXISTS: a real CUDA OOM during a model load cannot be produced on
every host (WSL's WDDM spills GPU memory into system RAM instead of failing),
so this reproduces the state such a load leaves behind. CTranslate2's
cuda_malloc_async allocator allocates with cudaMallocAsync and frees with
cudaFreeAsync on its own stream; with the default pool release threshold of 0
the memory goes back to the driver only at the next synchronisation, and after
a failed load nothing synchronises. The control half proves the memory really
is held, so a pass is not an artefact of the driver releasing it anyway.
"""

import ctypes
import glob
import os
import sys

sys.path.insert(0, os.environ.get("SUBGEN_TREE", "/work"))

import subgen  # noqa: E402

HOLD_MIB = 2048
CHUNK_MIB = 512
RESERVED_MEM_CURRENT = 5  # cudaMemPoolAttrReservedMemCurrent

libs = glob.glob(
    "/usr/local/lib/python3*/dist-packages/nvidia/cuda_runtime/lib/libcudart.so*"
)
if not libs:
    print("SKIP: no libcudart in this image")
    sys.exit(2)
cudart = ctypes.CDLL(libs[0])


def pool_reserved_mib():
    """MiB the device's default memory pool holds from the driver.

    The pool's own counter, not device free memory: other processes on the
    card (Ollama loading or unloading) move device free memory during the run,
    and a smoke reading that proxy fails or passes on their timing.
    """
    pool = ctypes.c_void_p()
    rc = cudart.cudaDeviceGetDefaultMemPool(ctypes.byref(pool), 0)
    if rc != 0:
        print(f"SKIP: cudaDeviceGetDefaultMemPool returned {rc} (no usable GPU)")
        sys.exit(2)
    reserved = ctypes.c_uint64()
    assert (
        cudart.cudaMemPoolGetAttribute(
            pool, RESERVED_MEM_CURRENT, ctypes.byref(reserved)
        )
        == 0
    )
    return reserved.value >> 20


def leave_pool_holding_memory():
    """Allocate like a model load, then free asynchronously and never sync."""
    stream = ctypes.c_void_p()
    assert cudart.cudaStreamCreate(ctypes.byref(stream)) == 0
    ptrs = []
    for _ in range(HOLD_MIB // CHUNK_MIB):
        p = ctypes.c_void_p()
        rc = cudart.cudaMallocAsync(
            ctypes.byref(p), ctypes.c_size_t(CHUNK_MIB << 20), stream
        )
        assert rc == 0, f"cudaMallocAsync returned {rc}"
        ptrs.append(p)
    assert cudart.cudaStreamSynchronize(stream) == 0
    for p in ptrs:
        assert cudart.cudaFreeAsync(p, stream) == 0


if pool_reserved_mib() != 0:
    print(
        f"SKIP: the default pool already holds {pool_reserved_mib()} MiB before the test"
    )
    sys.exit(2)

leave_pool_holding_memory()
held = pool_reserved_mib()
print(
    f"control: pool holds {held} MiB after async free with no sync (expect {HOLD_MIB})"
)

result = subgen.release_cuda_pool_after_failed_load()
left = pool_reserved_mib()
print(
    f"after release_cuda_pool_after_failed_load(): returned {result}, pool holds {left} MiB"
)

fails = []
if held < HOLD_MIB:
    fails.append(
        f"control did not hold the memory ({held} MiB): the scenario was not reproduced"
    )
if result != (held, 0):
    fails.append(f"helper reported {result}, expected ({held}, 0)")
if left != 0:
    fails.append(f"{left} MiB still held after the helper")

print("RESULT:", "PASSED" if not fails else "FAILED: " + "; ".join(fails))
sys.exit(1 if fails else 0)
