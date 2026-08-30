"""The per-video report.

Written beside the subtitles when reports are switched on. Its job is to answer
"why does this file have gaps?" without anyone reading Python or a log.

Plain text rather than JSON on purpose: this is read by a person looking at a
file that came out wrong, not parsed by anything.

The two numbers that matter most are near the top:

  speech detected   - what survived voice detection. If a talkative file says
                      5%, the dialogue was thrown away BEFORE Whisper saw it,
                      and no amount of Whisper tuning will recover it.
  mean confidence   - how sure the model was. Consistently low means it heard
                      something and struggled; the transcript is guesswork.

Those point at different fixes, which is the whole reason both are here.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from . import languages
from .profiles import DOUBTFUL_LOGPROB, Diagnostics, Profile, more_permissive_than
from .srt import Cue

# Listing every doubtful cue in a two-hour file would bury the summary.
MAX_LISTED_CUES = 40


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _pct(value: float | None) -> str:
    return "unknown" if value is None else f"{value * 100:.0f}%"


def _speech_comment(fraction: float | None, profile: Profile) -> list[str]:
    """Say what the speech percentage probably means.

    Deliberately hedged. Sparse dialogue and over-aggressive voice detection
    produce the same number, and only the person who knows the material can
    tell them apart. Saying which is which would be inventing a fact.
    """
    if fraction is None:
        return []
    if fraction >= 0.5:
        return []
    # Suggest the NEXT profile up, never the one already in use.
    nxt = more_permissive_than(profile.key)
    advice = (
        f"  try the '{nxt.name}' profile on this folder."
        if nxt is not None
        else "  though this folder already uses the most permissive profile, so the"
             " limit is the recording rather than the settings."
    )
    if fraction >= 0.15:
        return [
            "  Most of this file was judged to be silence. That is normal for",
            "  sparse dialogue, and a problem if the file is talkative - in which",
            "  case voice detection is discarding speech before Whisper sees it:",
            advice,
        ]
    return [
        "  Very little of this file was judged to be speech. If that does not",
        "  match the material, voice detection is the cause of any gaps, not",
        "  Whisper -",
        advice,
    ]


def _confidence_comment(mean: float | None) -> list[str]:
    if mean is None:
        return []
    if mean >= DOUBTFUL_LOGPROB:
        return []
    return [
        "  Average confidence is low. The model heard something and struggled",
        "  with it, so expect wrong words rather than missing ones. A better",
        "  source file helps more than any setting here.",
    ]


def render(
    *,
    video: Path,
    cues: list[Cue],
    diagnostics: Diagnostics,
    profile: Profile,
    model: str,
    device: str,
    compute_type: str,
    language: str | None,
    detected_language: str | None,
    language_probability: float | None,
    rule_name: str | None = None,
    source: str = "transcribed",
    source_detail: str | None = None,
    now: datetime | None = None,
) -> str:
    when = (now or datetime.now()).isoformat(timespec="seconds")
    out: list[str] = [
        f"subwright report for {video.name}",
        f"generated {when}",
        "",
    ]

    out.append("SOURCE")
    if source != "transcribed":
        out += [
            f"  subtitles were REUSED, not generated ({source})",
            f"  from: {source_detail or 'unknown'}",
            "",
            "  No transcription ran, so there are no confidence figures below.",
            "",
        ]
    else:
        out += [
            f"  transcribed on {device} ({compute_type}) with model {model}",
            f"  audio profile: {profile.name} - {profile.summary}",
        ]
        if rule_name:
            out.append(f"  watch folder: {rule_name}")
        if language:
            out.append(f"  language: {languages.name(language)} (pinned)")
        else:
            confidence = ("" if language_probability is None
                          else f", {language_probability * 100:.0f}% confident")
            out.append(
                f"  language: {languages.name(detected_language)} "
                f"(auto-detected{confidence})"
            )
        out.append("")

    out.append("AUDIO")
    out += [
        f"  media duration    {_hms(diagnostics.duration)}",
        f"  speech detected   {_hms(diagnostics.duration_after_vad)}"
        f"  ({_pct(diagnostics.speech_fraction)} of the file)",
        f"  silence removed   {_hms(diagnostics.duration - diagnostics.duration_after_vad)}",
    ]
    comment = _speech_comment(diagnostics.speech_fraction, profile)
    if comment:
        out += ["", *comment]
    out.append("")

    out.append("RESULT")
    out.append(f"  cues written      {len(cues)}")
    if source == "transcribed":
        mean = diagnostics.mean_logprob
        out.append(
            "  mean confidence   "
            + ("unknown" if mean is None else f"{mean:.2f}   (0 is perfect; below "
                                              f"{DOUBTFUL_LOGPROB} is doubtful)")
        )
        doubtful = diagnostics.doubtful_count
        share = f" ({doubtful / len(cues) * 100:.0f}% of cues)" if cues else ""
        out.append(f"  low confidence    {doubtful} cues{share}")
        comment = _confidence_comment(mean)
        if comment:
            out += ["", *comment]
    out.append("")

    if source == "transcribed":
        out.append("SETTINGS USED")
        for label, value in profile.describe():
            out.append(f"  {label:<28}{value}")
        out.append("")

        doubtful_cues = [c for c in cues
                         if c.avg_logprob is not None and c.avg_logprob < DOUBTFUL_LOGPROB]
        if doubtful_cues:
            out.append(f"LEAST CONFIDENT LINES ({len(doubtful_cues)} total)")
            out.append("  These are the ones worth checking against the audio.")
            out.append("")
            worst = sorted(doubtful_cues, key=lambda c: c.avg_logprob or 0)[:MAX_LISTED_CUES]
            for cue in sorted(worst, key=lambda c: c.start):
                text = " ".join(cue.text.split())[:70]
                out.append(f"  {_hms(cue.start)}  {cue.avg_logprob:6.2f}  {text}")
            if len(doubtful_cues) > MAX_LISTED_CUES:
                out.append(f"  ... and {len(doubtful_cues) - MAX_LISTED_CUES} more")
            out.append("")

    return "\n".join(out).rstrip() + "\n"
