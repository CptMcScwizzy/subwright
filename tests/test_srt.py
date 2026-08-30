"""Tests for SRT rendering.

Test names state the behaviour in English so the pass list is readable without
knowing Python. See docs/CONTRACT.md.
"""

from pathlib import Path

from subwright.srt import MAX_CUE_SECONDS, Cue, cap_cue, format_timestamp, normalise_text, render

DATA = Path(__file__).parent / "data"


# --- timestamps ---

def test_timestamp_zero_is_all_zeroes():
    assert format_timestamp(0) == "00:00:00,000"


def test_timestamp_formats_hours_minutes_seconds_millis():
    # 1h 2m 3.456s
    assert format_timestamp(3723.456) == "01:02:03,456"


def test_timestamp_truncates_milliseconds_rather_than_rounding():
    # 0.9999s is 999ms, not 1000ms. Frozen from the original implementation:
    # rounding would shift every timestamp in every previously generated file.
    assert format_timestamp(0.9999) == "00:00:00,999"


def test_timestamp_never_goes_negative():
    assert format_timestamp(-5) == "00:00:00,000"


# --- cue capping ---

def test_cue_longer_than_five_seconds_is_capped():
    capped = cap_cue(Cue(start=10.0, end=99.0, text="x"))
    assert capped.end == 10.0 + MAX_CUE_SECONDS


def test_cue_shorter_than_cap_is_unchanged():
    capped = cap_cue(Cue(start=10.0, end=12.0, text="x"))
    assert capped.end == 12.0


def test_cue_exactly_at_cap_is_unchanged():
    capped = cap_cue(Cue(start=0.0, end=MAX_CUE_SECONDS, text="x"))
    assert capped.end == MAX_CUE_SECONDS


def test_cue_end_never_precedes_or_equals_start():
    capped = cap_cue(Cue(start=5.0, end=5.0, text="x"))
    assert capped.end > capped.start


# --- text handling ---

def test_embedded_blank_lines_are_removed():
    # A blank line inside a cue truncates the subtitle in most players.
    assert normalise_text("first\n\nsecond") == "first\nsecond"


def test_surrounding_whitespace_is_stripped():
    assert normalise_text("  hello  ") == "hello"


# --- rendering ---

def test_numbering_is_contiguous_when_empty_cues_are_dropped():
    cues = [
        Cue(0.0, 1.0, "one"),
        Cue(1.0, 2.0, "   "),      # dropped
        Cue(2.0, 3.0, "three"),
    ]
    out = render(cues)
    assert out.startswith("1\n")
    assert "\n2\n" in out
    assert "\n3\n" not in out          # only two cues survive
    assert "three" in out


def test_empty_cues_produce_no_output_at_all():
    assert render([Cue(0.0, 1.0, "")]) == ""


def test_renders_golden_file_byte_for_byte():
    """The regression net for the whole output format.

    If this fails, the subtitle format changed - check that was intended before
    updating the golden file.
    """
    cues = [
        Cue(0.0, 2.5, "Hello there."),
        Cue(2.5, 60.0, "This one runs long and gets capped."),
        Cue(61.25, 62.5, "  Padded  "),
        Cue(63.0, 64.0, ""),
        Cue(3723.456, 3724.0, "Late in the file."),
    ]
    assert render(cues) == (DATA / "golden_basic.srt").read_text(encoding="utf-8")


def test_output_has_no_carriage_returns():
    out = render([Cue(0.0, 1.0, "a")])
    assert "\r" not in out
