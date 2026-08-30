"""Finding subtitles that already exist, so the GPU is not asked twice.

Two sources, checked in this order:

1. A sidecar `.srt` sitting next to the video in the drop folder. Someone
   deliberately put it there.
2. An English subtitle track already inside the video file.

Extracting either takes about a second. Transcribing the same film takes
minutes of GPU on a card shared with other things. On a library where a good
share of files already carry subtitles, this is the single largest saving
available.

The rule throughout: reuse is an OPTIMISATION. Anything unexpected - a
zero-byte sidecar, ffprobe missing, an extraction that fails - falls back to
transcribing. It must never turn a file that would have worked into one that
does not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .mediaprobe import Prober, ProbeUnavailable, choose_english_stream

log = logging.getLogger(__name__)

# Sidecar naming, in preference order. Plex and friends write the middle ones.
SIDECAR_SUFFIXES = (".srt", ".en.srt", ".eng.srt", ".english.srt")

# A file smaller than this cannot hold a usable subtitle. Guards against the
# zero-byte and half-written sidecars that turn up in download folders.
MIN_SIDECAR_BYTES = 32


@dataclass(frozen=True)
class Reuse:
    """Subtitles that already exist and can be used as they are."""

    kind: str          # "sidecar" or "embedded"
    detail: str        # human-readable, for the log and the history page
    sidecar: Path | None = None
    stream_index: int | None = None


def looks_like_srt(text: str) -> bool:
    """Cheapest possible sanity check: does it contain a cue timing?

    Deliberately not a parser. The question is "did we get a subtitle file or
    something else", and an arrow between two timestamps answers it. Being
    stricter would start rejecting real files over formatting trivia.
    """
    return " --> " in text


def _readable_srt(path: Path) -> bool:
    try:
        if path.stat().st_size < MIN_SIDECAR_BYTES:
            log.info("ignoring %s: too small to be a subtitle file", path.name)
            return False
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("could not read %s: %s", path.name, exc)
        return False
    if not looks_like_srt(text):
        log.info("ignoring %s: does not look like an SRT file", path.name)
        return False
    return True


def find_sidecar(video: Path) -> Path | None:
    """A usable `.srt` beside the video, or None.

    Matched on the video's stem, so `Foo.mkv` finds `Foo.srt` and `Foo.en.srt`
    but never `Bar.srt`.
    """
    for suffix in SIDECAR_SUFFIXES:
        candidate = video.with_name(f"{video.stem}{suffix}")
        if candidate.is_file() and _readable_srt(candidate):
            return candidate
    return None


def find_embedded(video: Path, prober: Prober) -> Reuse | None:
    """An English text subtitle track inside the video, or None."""
    try:
        streams = prober.subtitle_streams(video)
    except ProbeUnavailable as exc:
        # Not an error for the job: it just means we transcribe, as before.
        log.info("could not inspect %s for subtitles: %s", video.name, exc)
        return None

    if not streams:
        return None

    chosen = choose_english_stream(streams)
    if chosen is None:
        log.info(
            "%s has %d subtitle track(s) but none usable (%s)",
            video.name, len(streams), "; ".join(s.describe() for s in streams[:4]),
        )
        return None

    return Reuse(kind="embedded", detail=chosen.describe(), stream_index=chosen.index)


def find(video: Path, prober: Prober | None) -> Reuse | None:
    """Subtitles already available for this video, or None.

    A sidecar wins over an embedded track: someone put it there on purpose,
    which is a stronger signal than whatever the muxer happened to include.
    """
    sidecar = find_sidecar(video)
    if sidecar is not None:
        return Reuse(kind="sidecar", detail=sidecar.name, sidecar=sidecar)
    if prober is None:
        return None
    return find_embedded(video, prober)


def sidecars_for(video: Path) -> list[Path]:
    """Every sidecar belonging to this video, whether usable or not.

    Used when a video is moved out of the drop folder: leaving its subtitles
    behind would strand them there, which is what used to happen.
    """
    out = []
    for suffix in SIDECAR_SUFFIXES:
        candidate = video.with_name(f"{video.stem}{suffix}")
        if candidate.is_file() and candidate not in out:
            out.append(candidate)
    return out
