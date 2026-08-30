"""Reading and extracting subtitle tracks already inside a video file.

THE ONLY MODULE THAT SHELLS OUT TO ffprobe/ffmpeg for subtitles.

Same arrangement as `transcriber`: everything else depends on the `Prober`
protocol below, so stream selection - which is where all the judgement lives -
is tested against canned ffprobe output with no media files and no ffmpeg. This
is the second and last external-binary seam in the application; resist adding a
third.

Extracting a track that is already there takes about a second. Transcribing the
same film takes minutes of GPU. That is the entire justification for this file.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)

# ffprobe reads headers, and ffmpeg's subtitle extraction does no re-encoding,
# so both are fast. The timeouts exist to stop a damaged file or a stalled NFS
# mount wedging the worker thread forever, not to bound normal work.
PROBE_TIMEOUT = 60
EXTRACT_TIMEOUT = 300

# Subtitles that are text and can therefore become SRT.
TEXT_CODECS = frozenset({"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"})

# Subtitles that are pictures. Converting these to SRT needs OCR, which this
# application does not do and should not start doing. Recognised explicitly so
# they are SKIPPED rather than producing an empty file.
IMAGE_CODECS = frozenset({
    "hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub", "dvbsub", "pgssub",
})

ENGLISH_TAGS = frozenset({"en", "eng", "english", "en-us", "en-gb"})


@dataclass(frozen=True)
class SubtitleStream:
    index: int
    codec: str
    language: str | None = None
    title: str | None = None
    forced: bool = False
    hearing_impaired: bool = False
    default: bool = False

    @property
    def is_text(self) -> bool:
        return self.codec.lower() in TEXT_CODECS

    @property
    def is_english(self) -> bool:
        if self.language and self.language.lower() in ENGLISH_TAGS:
            return True
        # Some muxers leave the language tag empty and put it in the title.
        return bool(self.title and "english" in self.title.lower())

    def describe(self) -> str:
        bits = [f"stream {self.index}", self.codec]
        if self.language:
            bits.append(self.language)
        if self.title:
            bits.append(repr(self.title))
        if self.hearing_impaired:
            bits.append("SDH")
        return ", ".join(bits)


class Prober(Protocol):
    def subtitle_streams(self, path: Path) -> list[SubtitleStream]: ...

    def extract(self, path: Path, index: int, target: Path) -> None: ...


class ProbeUnavailable(RuntimeError):
    """ffprobe or ffmpeg could not be run, or failed on this file."""


def choose_english_stream(streams: list[SubtitleStream]) -> SubtitleStream | None:
    """Pick the track worth extracting, or None.

    The rejections matter more than the ranking:

    Image-based tracks are skipped because turning them into SRT requires OCR.
    Extracting one anyway produces a file that looks valid and contains nothing.

    Forced tracks are skipped because they are not subtitles for the film - they
    are the handful of lines where a character speaks a foreign language in an
    otherwise English film. Reusing one would leave almost the whole film
    unsubtitled while marking the job done, which is worse than not reusing at
    all.

    Non-English tracks are skipped because the output of this application is
    English. A Japanese track is the input, not the answer.
    """
    usable = [
        s for s in streams
        if s.is_text and s.is_english and not s.forced
    ]
    if not usable:
        return None

    # Prefer a plain track over one written for the hearing impaired: SDH
    # interleaves "[door creaks]" with the dialogue. Accepted if it is all
    # there is - it is still a real transcript.
    def rank(s: SubtitleStream) -> tuple:
        return (s.hearing_impaired, not s.default, s.index)

    return sorted(usable, key=rank)[0]


def _disposition(raw: dict, key: str) -> bool:
    return bool((raw.get("disposition") or {}).get(key))


def parse_streams(payload: dict) -> list[SubtitleStream]:
    """Turn ffprobe JSON into SubtitleStreams. Pure, so it is directly testable."""
    out = []
    for raw in payload.get("streams") or []:
        if raw.get("codec_type") != "subtitle":
            continue
        tags = raw.get("tags") or {}
        index = raw.get("index")
        if index is None:
            continue
        out.append(SubtitleStream(
            index=int(index),
            codec=str(raw.get("codec_name") or "").lower(),
            language=tags.get("language") or tags.get("LANGUAGE"),
            title=tags.get("title") or tags.get("TITLE"),
            forced=_disposition(raw, "forced"),
            hearing_impaired=_disposition(raw, "hearing_impaired"),
            default=_disposition(raw, "default"),
        ))
    return out


class FfmpegProber:
    """The real thing. Shells out; never imported by the tests."""

    def __init__(self, ffprobe: str = "ffprobe", ffmpeg: str = "ffmpeg") -> None:
        self.ffprobe = ffprobe
        self.ffmpeg = ffmpeg

    def subtitle_streams(self, path: Path) -> list[SubtitleStream]:
        cmd = [
            self.ffprobe, "-v", "error",
            "-select_streams", "s",
            "-show_entries", "stream=index,codec_name,codec_type:stream_tags:stream_disposition",
            "-of", "json", str(path),
        ]
        try:
            done = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=PROBE_TIMEOUT, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProbeUnavailable(f"ffprobe failed on {path.name}: {exc}") from exc
        if done.returncode != 0:
            raise ProbeUnavailable(
                f"ffprobe exited {done.returncode} on {path.name}: {done.stderr.strip()[:300]}"
            )
        try:
            return parse_streams(json.loads(done.stdout or "{}"))
        except json.JSONDecodeError as exc:
            raise ProbeUnavailable(f"ffprobe output was not JSON for {path.name}") from exc

    def extract_command(self, path: Path, index: int, target: Path) -> list[str]:
        """The ffmpeg invocation, built separately so it can be asserted on.

        `-f srt` is not optional. ffmpeg picks the output format from the file
        extension, and the target here is a scratch file ending `.srt.tmp` -
        which it does not recognise, so without this it fails with "Unable to
        find a suitable output format". The scratch name is not negotiable
        either: the write has to be atomic.

        `-c:s srt` rather than `copy` because ASS and mov_text need converting,
        and converting an already-SRT track costs nothing.
        """
        return [
            self.ffmpeg, "-nostdin", "-v", "error", "-y",
            "-i", str(path),
            "-map", f"0:{index}",
            "-c:s", "srt",
            "-f", "srt",
            str(target),
        ]

    def extract(self, path: Path, index: int, target: Path) -> None:
        """Write stream `index` of `path` to `target` as SRT."""
        cmd = self.extract_command(path, index, target)
        try:
            done = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=EXTRACT_TIMEOUT, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProbeUnavailable(f"ffmpeg failed on {path.name}: {exc}") from exc
        if done.returncode != 0:
            raise ProbeUnavailable(
                f"ffmpeg exited {done.returncode} extracting stream {index} "
                f"from {path.name}: {done.stderr.strip()[:300]}"
            )
