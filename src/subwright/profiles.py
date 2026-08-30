"""Audio profiles: how hard to work at hearing speech.

Whisper can discard dialogue at two independent stages, and both are silent
about it:

1. Voice detection (Silero) runs BEFORE transcription. Anything it does not
   mark as speech is cut out and Whisper never sees it. Reverb tails, distant
   microphones and quiet delivery are exactly what it misjudges.
2. Whisper's own segment filters then throw away results that look like
   silence, or look repetitive enough to be hallucinated.

On clean audio the defaults are right. On echoey, noisy or amateur recordings
they remove real speech, which shows up as gaps. These profiles move both
stages together, because moving one without the other does very little.

**The trade-off is real and runs the other way too.** A more permissive profile
means more hallucination: Whisper will write fluent, confident, entirely
invented dialogue over noise. There is no setting that only finds true
positives. That is why this is chosen per watch folder rather than globally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Whisper's own defaults, for reference in the table below:
#   vad threshold 0.5, no_speech_threshold 0.6, log_prob_threshold -1.0,
#   compression_ratio_threshold 2.4


@dataclass(frozen=True)
class Profile:
    key: str
    name: str
    summary: str
    vad_threshold: float
    vad_min_silence_ms: int
    vad_speech_pad_ms: int
    no_speech_threshold: float
    log_prob_threshold: float
    compression_ratio_threshold: float
    condition_on_previous_text: bool = True
    beam_size: int = 5

    def transcribe_options(self) -> dict[str, Any]:
        """The kwargs handed to faster-whisper.

        `task` is not here and never will be: Whisper translates into English
        and nothing else, so it is not a choice.
        """
        return {
            "task": "translate",
            "beam_size": self.beam_size,
            "vad_filter": True,
            "vad_parameters": {
                "threshold": self.vad_threshold,
                "min_silence_duration_ms": self.vad_min_silence_ms,
                "speech_pad_ms": self.vad_speech_pad_ms,
            },
            "no_speech_threshold": self.no_speech_threshold,
            "log_prob_threshold": self.log_prob_threshold,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "condition_on_previous_text": self.condition_on_previous_text,
        }

    def describe(self) -> list[tuple[str, Any]]:
        """Flat (label, value) pairs for the report file."""
        return [
            ("profile", f"{self.key} ({self.name})"),
            ("beam size", self.beam_size),
            ("vad threshold", self.vad_threshold),
            ("vad min silence", f"{self.vad_min_silence_ms} ms"),
            ("vad speech padding", f"{self.vad_speech_pad_ms} ms"),
            ("no-speech threshold", self.no_speech_threshold),
            ("log-prob threshold", self.log_prob_threshold),
            ("compression ratio threshold", self.compression_ratio_threshold),
            ("condition on previous text", self.condition_on_previous_text),
        ]


STANDARD = Profile(
    key="standard",
    name="Standard",
    summary="Well-recorded audio: studio, broadcast, a decent microphone.",
    vad_threshold=0.35,
    vad_min_silence_ms=300,
    vad_speech_pad_ms=200,
    # 0.6 is Whisper's default. The script this replaced used 0.4 with the
    # comment "lower: less likely to drop quiet passages", which is backwards -
    # a segment is skipped when no_speech_prob EXCEEDS this, so lowering it
    # discards MORE. See faster_whisper/transcribe.py, "no voice activity
    # check". That is a bug, and it is fixed here rather than preserved.
    no_speech_threshold=0.6,
    log_prob_threshold=-1.0,
    # Kept from the original: 2.4 rejects genuinely repetitive dialogue.
    compression_ratio_threshold=2.8,
)

DIFFICULT = Profile(
    key="difficult",
    name="Difficult audio",
    summary=(
        "Echoey, noisy, quiet or distant recordings. Hears more, and invents "
        "more over background noise."
    ),
    # Quieter speech still counts as speech.
    vad_threshold=0.20,
    # Longer, so a reverb tail does not split one line into fragments that are
    # each then judged too short to be speech.
    vad_min_silence_ms=500,
    # Wider, so the first syllable after a pause is not clipped off.
    vad_speech_pad_ms=400,
    # Harder to declare a segment silent...
    no_speech_threshold=0.8,
    # ...and the low-confidence rescue fires more readily. This pair matters
    # most: poor audio produces low confidence by definition, which is exactly
    # where the default refuses to rescue.
    log_prob_threshold=-1.5,
    compression_ratio_threshold=3.0,
)

MAXIMUM = Profile(
    key="maximum",
    name="Maximum recall",
    summary=(
        "Keeps almost everything. Expect invented dialogue in silent stretches "
        "- use it to find what the other profiles miss, not as a default."
    ),
    vad_threshold=0.10,
    vad_min_silence_ms=700,
    vad_speech_pad_ms=600,
    no_speech_threshold=0.95,
    log_prob_threshold=-2.0,
    compression_ratio_threshold=3.4,
    # On poor audio, conditioning on previous text is what sends Whisper into
    # repetition loops - it keeps agreeing with its own mistake. Off here.
    condition_on_previous_text=False,
)

PROFILES: dict[str, Profile] = {p.key: p for p in (STANDARD, DIFFICULT, MAXIMUM)}
DEFAULT_PROFILE = STANDARD.key

# Ordered least to most permissive, so advice can name the next step up rather
# than recommending the profile already in use.
LADDER = [STANDARD.key, DIFFICULT.key, MAXIMUM.key]

# An average log probability below this is worth a second look. Not a hard
# threshold anywhere - it only decides what the report highlights.
DOUBTFUL_LOGPROB = -1.0


def more_permissive_than(key: str | None) -> Profile | None:
    """The next profile up, or None if this is already the most permissive."""
    try:
        index = LADDER.index(key or DEFAULT_PROFILE)
    except ValueError:
        return PROFILES[DIFFICULT.key]
    if index + 1 >= len(LADDER):
        return None
    return PROFILES[LADDER[index + 1]]


def get(key: str | None) -> Profile:
    """Look up a profile, falling back to Standard rather than failing.

    A job must not die because a profile was renamed or a setting arrived from
    an older database.
    """
    if not key:
        return PROFILES[DEFAULT_PROFILE]
    return PROFILES.get(key, PROFILES[DEFAULT_PROFILE])


def is_valid(key: str) -> bool:
    return key in PROFILES


def choices() -> list[tuple[str, str, str]]:
    """(key, name, summary) for the settings page."""
    return [(p.key, p.name, p.summary) for p in PROFILES.values()]


@dataclass
class Diagnostics:
    """What a finished job can say about how well it heard the audio."""

    duration: float = 0.0
    duration_after_vad: float = 0.0
    cue_count: int = 0
    logprobs: list[float] = field(default_factory=list)

    @property
    def speech_fraction(self) -> float | None:
        if self.duration <= 0:
            return None
        return max(0.0, min(1.0, self.duration_after_vad / self.duration))

    @property
    def mean_logprob(self) -> float | None:
        if not self.logprobs:
            return None
        return sum(self.logprobs) / len(self.logprobs)

    @property
    def doubtful_count(self) -> int:
        return sum(1 for v in self.logprobs if v < DOUBTFUL_LOGPROB)
