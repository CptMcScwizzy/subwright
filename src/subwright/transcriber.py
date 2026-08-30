"""Speech-to-subtitle transcription.

THE ONLY MODULE THAT IMPORTS faster_whisper.

Everything else depends on the `Transcriber` protocol below, which is why the
whole pipeline - collisions, markers, resume, atomic writes, signals - is
testable with no GPU, no model download and no video files. Substituting
FakeTranscriber in tests exercises the real production code path; only this
module needs hardware.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .profiles import DEFAULT_PROFILE, Profile, get
from .srt import Cue

log = logging.getLogger(__name__)

# Hand-tuned against real material and deliberately NOT user-configurable.
# Exposing them would add settings nobody can meaningfully evaluate and seven
# untested code paths. Changing one is a commit and a version bump, which is
# reviewable and revertible. Frozen from the original script.
# Kept as the Standard profile's options, so the contract test that asserts
# the tuning has not drifted still has one thing to compare against. The live
# values now come from profiles.py, which is where they can vary per folder.
STANDARD_OPTIONS: dict = {
    # X -> English. Whisper can only translate INTO English; this is the whole
    # point of the application.
    "task": "translate",
    "beam_size": 5,
    # Voice activity detection: skips silence, which on sparse audio is most of
    # the file. Without it, Whisper hallucinates text over long silences.
    "vad_filter": True,
    "vad_parameters": {
        "min_silence_duration_ms": 300,  # shorter than default: catches quick gaps
        "speech_pad_ms": 200,            # avoids clipping the start of speech
        "threshold": 0.3,                # lower than default: catches quiet speech
    },
    "no_speech_threshold": 0.4,          # lower: less likely to drop quiet passages
    "compression_ratio_threshold": 2.8,  # higher: more lenient about repetition
    "condition_on_previous_text": True,  # better continuity across segments
}

TRANSCRIBE_OPTIONS = STANDARD_OPTIONS  # backwards-compatible alias


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    detected_language: str | None = None
    language_probability: float | None = None
    # How much audio survived voice detection. The single most useful number
    # for diagnosing missing dialogue: if a talkative file reports 5%, the
    # speech was discarded before Whisper ever saw it, and no amount of
    # Whisper tuning will bring it back.
    duration_after_vad: float = 0.0


class Transcriber(Protocol):
    """One method. Implemented for real below, and faked in tests."""

    def transcribe(
        self, path: Path, language: str | None, profile: Profile | None = None,
    ) -> tuple[Iterator[Cue], MediaInfo]:
        ...


class TranscriberUnavailable(RuntimeError):
    """Raised when the requested device cannot be used.

    Deliberately fatal when device=cuda. The original silently fell back to CPU
    and ran 20-50x slower with one WARNING line, which is indistinguishable from
    'the GPU is just busy' until you check nvidia-smi hours later.
    """


class FasterWhisperTranscriber:
    """faster-whisper, loaded once and reused."""

    def __init__(
        self,
        model_size: str,
        *,
        device: str = "cuda",
        compute_type: str = "int8",
        download_root: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._download_root = download_root
        self._model = None

    def load(self):
        """Load the model. Raises TranscriberUnavailable if the device is not usable."""
        if self._model is not None:
            return self._model
        from faster_whisper import WhisperModel  # imported late: heavy, and GPU-bound

        started = time.monotonic()
        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self._download_root,
            )
        except Exception as exc:
            if self.device == "cpu":
                raise
            raise TranscriberUnavailable(
                f"could not load {self.model_size} on {self.device} "
                f"with compute_type={self.compute_type}: {exc}"
            ) from exc
        log.info(
            "loaded %s on %s (%s) in %.1fs",
            self.model_size, self.device, self.compute_type, time.monotonic() - started,
        )
        return self._model

    def transcribe(
        self, path: Path, language: str | None, profile: Profile | None = None,
    ) -> tuple[Iterator[Cue], MediaInfo]:
        model = self.load()
        chosen = profile or get(DEFAULT_PROFILE)
        opts = chosen.transcribe_options()
        if language:
            opts["language"] = language
        log.info("transcribing %s with the %s profile", path.name, chosen.key)
        segments, info = model.transcribe(str(path), **opts)

        def cues() -> Iterator[Cue]:
            for seg in segments:
                try:
                    yield Cue(
                        start=seg.start, end=seg.end, text=seg.text,
                        avg_logprob=getattr(seg, "avg_logprob", None),
                        no_speech_prob=getattr(seg, "no_speech_prob", None),
                        compression_ratio=getattr(seg, "compression_ratio", None),
                    )
                except (AttributeError, IndexError) as exc:
                    # One malformed segment should not lose the whole file.
                    log.warning("skipping malformed segment: %s", exc)
                    continue

        duration = getattr(info, "duration", 0.0) or 0.0
        return cues(), MediaInfo(
            duration=duration,
            detected_language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            # Older faster-whisper builds omit it; fall back to the full
            # duration so the report says "all of it" rather than "none".
            duration_after_vad=getattr(info, "duration_after_vad", duration) or duration,
        )
