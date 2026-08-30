"""SubRip (.srt) rendering.

Pure functions only - no I/O, no clock, no filesystem. Everything here is
directly testable, and the golden-file test in tests/test_srt.py is the
regression net for the entire output format.

Behaviour frozen from the original translate_watcher.py:
  - timestamps are HH:MM:SS,mmm with milliseconds TRUNCATED, not rounded
  - every cue is hard-capped at MAX_CUE_SECONDS
Both are deliberate. See docs/CONTRACT.md.
"""

from __future__ import annotations

from dataclasses import dataclass

# A cue longer than this is truncated. Whisper occasionally emits a single
# segment spanning minutes when it loses the thread; without this cap one bad
# segment parks a line of text on screen for the rest of the scene.
MAX_CUE_SECONDS = 5.0

# SRT requires start < end. Zero-length cues are silently dropped by some
# players and rendered forever by others, so clamp to at least 1ms.
MIN_CUE_SECONDS = 0.001


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str
    # Whisper's own confidence for the segment this came from. Optional, and
    # ignored entirely by rendering - an SRT file has nowhere to put them.
    # They live here rather than in a parallel list because a list that has to
    # stay index-aligned with another list is a bug waiting to happen, and
    # these are only ever read alongside their cue.
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


def format_timestamp(seconds: float) -> str:
    """Seconds -> 'HH:MM:SS,mmm'.

    Milliseconds are truncated rather than rounded. This matches the original
    implementation exactly; "fixing" it to round would shift every timestamp in
    every previously generated subtitle by up to a millisecond, which would make
    old and new output incomparable during the migration for no benefit.
    """
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def normalise_text(text: str) -> str:
    """Collapse a cue's text to something SRT can represent.

    A blank line inside a cue terminates it early in most players, silently
    truncating the subtitle. Whisper does occasionally emit these.
    """
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(line for line in lines if line).strip()


def cap_cue(cue: Cue) -> Cue:
    """Clamp a cue to MAX_CUE_SECONDS and guarantee end > start."""
    start = max(0.0, cue.start)
    end = cue.end
    if end - start > MAX_CUE_SECONDS:
        end = start + MAX_CUE_SECONDS
    if end - start < MIN_CUE_SECONDS:
        end = start + MIN_CUE_SECONDS
    return Cue(start=start, end=end, text=cue.text)


def render(cues: list[Cue]) -> str:
    """Render cues to SRT text.

    Empty cues are dropped BEFORE numbering, so indices are contiguous. The
    original incremented its counter even when skipping an empty segment, which
    left gaps in the numbering - tolerated by most players but not valid SRT.
    """
    out: list[str] = []
    index = 0
    for cue in cues:
        text = normalise_text(cue.text)
        if not text:
            continue
        index += 1
        capped = cap_cue(cue)
        out.append(str(index))
        out.append(f"{format_timestamp(capped.start)} --> {format_timestamp(capped.end)}")
        out.append(text)
        out.append("")
    if not out:
        return ""
    # Trailing blank line: every cue block ends with one, including the last.
    # This matches the original byte-for-byte, which matters while old and new
    # output are being compared during the migration.
    return "\n".join(out) + "\n"
