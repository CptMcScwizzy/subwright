"""Test doubles.

FakeTranscriber implements the same Transcriber protocol as the real one, so
tests using it run the real jobs/worker code - only the model is substituted.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from subwright.srt import Cue
from subwright.transcriber import MediaInfo

DEFAULT_CUES = [
    Cue(0.0, 2.0, "First line."),
    Cue(2.0, 4.0, "Second line."),
    Cue(4.0, 30.0, "Third line, long enough to be capped."),
]


class FakeTranscriber:
    """Returns canned cues instantly. Can be told to fail, or to run a hook."""

    def __init__(
        self,
        cues: list[Cue] | None = None,
        *,
        raise_on_call: Exception | None = None,
        raise_after_cues: int | None = None,
        on_call=None,
        duration: float = 60.0,
    ) -> None:
        self._cues = DEFAULT_CUES if cues is None else cues
        self._raise_on_call = raise_on_call
        self._raise_after_cues = raise_after_cues
        self._on_call = on_call
        self._duration = duration
        self.calls: list[tuple[Path, str | None]] = []

    def transcribe(self, path: Path, language: str | None) -> tuple[Iterator[Cue], MediaInfo]:
        self.calls.append((path, language))
        if self._on_call is not None:
            # Lets a test assert on filesystem state at the moment transcription
            # begins - used to prove the video is moved BEFORE transcribing.
            self._on_call(path)
        if self._raise_on_call is not None:
            raise self._raise_on_call

        cues = self._cues
        limit = self._raise_after_cues

        def gen() -> Iterator[Cue]:
            for i, cue in enumerate(cues):
                if limit is not None and i >= limit:
                    raise RuntimeError("transcription died mid-file")
                yield cue

        return gen(), MediaInfo(duration=self._duration, detected_language="ja")
