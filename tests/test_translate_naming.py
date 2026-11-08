"""Patch 0067: a per-request task decides the subtitle's file name.

Before 0067, a POST /batch?task=translate on an instance whose global
TRANSCRIBE_OR_TRANSLATE is "transcribe" wrote English text into "<stem>.ko.srt",
because the naming only looked at the global and result.language is the SOURCE
language. These tests run the two naming functions from the patched tree with
plain globals (no torch / whisper needed), so they run in the PR workflow.
"""

import ast
import enum
import os
import pathlib
import sys

import pytest

UPSTREAM = pathlib.Path(__file__).resolve().parent.parent / "upstream"
SUBGEN = UPSTREAM / "subgen.py"
FUNCS = ("define_subtitle_language_naming", "name_subtitle")


def _load(**globals_):
    if not SUBGEN.exists():
        pytest.skip("run scripts/apply-patches.sh first")
    tree = ast.parse(SUBGEN.read_text(encoding="utf-8"))
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCS}
    if len(defs) != 2:
        pytest.skip("naming functions not found - run scripts/apply-patches.sh")
    # no skip when 0067 is missing: the per-request tests must FAIL on the old naming
    sys.path.insert(0, str(UPSTREAM))
    try:
        from language_code import LanguageCode
    finally:
        sys.path.remove(str(UPSTREAM))
    ns = {"os": os, "LanguageCode": LanguageCode, "Enum": enum.Enum,
          "subtitle_language_name": "", "transcribe_or_translate": "transcribe",
          "show_in_subname_subgen": True, "show_in_subname_model": True, "whisper_model": "small",
          "subtitle_language_naming_type": "ISO_639_1"}
    ns.update(globals_)
    exec(compile(ast.Module(body=[defs[f] for f in FUNCS], type_ignores=[]), str(SUBGEN), "exec"), ns)
    return ns


VIDEO = "/media/tv/Run BTS!/Season 4/Run BTS! - S04E01.mp4"
STEM = os.path.splitext(VIDEO)[0]


def korean(ns):
    return ns["LanguageCode"].from_string("ko")


def test_per_request_translate_on_a_transcribe_instance_is_named_english():
    ns = _load(transcribe_or_translate="transcribe")
    assert ns["name_subtitle"](VIDEO, korean(ns), "translate") == STEM + ".subgen.small.en.srt"


def test_per_request_transcribe_keeps_the_source_language():
    ns = _load(transcribe_or_translate="transcribe")
    assert ns["name_subtitle"](VIDEO, korean(ns), "transcribe") == STEM + ".subgen.small.ko.srt"


def test_per_request_transcribe_on_a_translate_instance_keeps_the_source_language():
    # the job wrote Korean text, so it must not be labelled English just because the global says translate
    ns = _load(transcribe_or_translate="translate")
    assert ns["name_subtitle"](VIDEO, korean(ns), "transcribe") == STEM + ".subgen.small.ko.srt"


@pytest.mark.parametrize("global_task,suffix", [("translate", "en"), ("transcribe", "ko")])
def test_no_task_keeps_the_old_global_behaviour(global_task, suffix):
    # should_skip_file's expected-output check calls name_subtitle without a task
    ns = _load(transcribe_or_translate=global_task)
    assert ns["name_subtitle"](VIDEO, korean(ns)) == STEM + f".subgen.small.{suffix}.srt"


@pytest.mark.parametrize("naming,expected", [("ISO_639_2_B", "eng"), ("NAME", "English")])
def test_other_naming_types_get_the_english_form(naming, expected):
    ns = _load(subtitle_language_naming_type=naming, show_in_subname_subgen=False, show_in_subname_model=False)
    assert ns["name_subtitle"](VIDEO, korean(ns), "translate") == f"{STEM}.{expected}.srt"


def test_subtitle_language_name_still_wins():
    ns = _load(subtitle_language_name="es", show_in_subname_subgen=False, show_in_subname_model=False)
    assert ns["name_subtitle"](VIDEO, korean(ns), "translate") == STEM + ".es.srt"
