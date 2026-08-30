# Changelog

## v0.1.0

First release. Replaces a single-file script run by a systemd unit against a
hand-built virtualenv, which had no backups, no tests, and a CUDA setup that
would silently fall back to the CPU if a Python minor version changed.

### What it does

- Watches folders and generates English subtitles with faster-whisper on an
  NVIDIA GPU.
- Web UI on port 8420: live progress, settings, job history, actions.
- Ships as a container image with the exact CUDA wheel versions known to work.

### Fixed from the original script

- **Silent job abandonment.** Subtitles were streamed to an open file, and the
  video was moved out of `ingest/` *before* transcription. A crash left a
  truncated `.srt`, no success marker, no error marker, and nothing ever
  rescanned. The file was lost. Now: atomic writes, a `.processing` claim, and
  resume on the next start.
- **`no_speech_threshold` was backwards.** Set to `0.4` with the comment
  "lower: less likely to drop quiet passages", but a segment is skipped when
  `no_speech_prob` *exceeds* the threshold — so it discarded more than Whisper's
  `0.6` default, not less. This was a direct cause of missing dialogue on
  imperfect audio.
- **A data-loss window in the remux repair path**, which unlinked before
  renaming.
- **Silent CPU fallback.** `device=cuda` now fails loudly instead of running
  20–50× slower with one warning line.
- **Unbounded `.srt.bak` accumulation.**
- **No SIGTERM handling.**
- **Non-contiguous subtitle numbering** when empty cues were filtered.
- **A nonsense ETA** that swung between eight hours and forty minutes on the
  same file, because voice detection removes an unpredictable share of the
  audio. Replaced with position-in-media, which is honest about what it
  measures.
- **Logs written to `$HOME` and never rotated.**

### Added

- **Multiple watch folders**, each with its own drop folder, output folder,
  source language and audio profile.
- **Audio profiles** — Standard, Difficult audio, Maximum recall — moving voice
  detection and Whisper's segment filters together. On a test file, Difficult
  audio recovered 671 cues against 557, with 20% of the audio surviving voice
  detection rather than 14%.
- **Reuse of existing subtitles.** A sidecar `.srt` or an embedded English
  track is used instead of transcribing: about a second, against several
  minutes of GPU. Picture-based, forced and non-English tracks are never
  reused, and every failure falls back to transcribing.
- **Per-video diagnostic reports** (off by default) saying how much audio was
  heard, how confident the model was, which settings were used, and the least
  confident lines.
- **Live subtitle preview and progress** on the dashboard.
- **Language auto-detection reporting**, including a warning below 75%
  confidence — a wrong language produces fluent invented dialogue rather than
  an error.
- **History actions**: redo in place, remove an entry, clear the history.

### Notes

- Output is English only. Whisper has no target-language setting and cannot
  translate into anything else.
- `large-v3-turbo` is deliberately absent: it cannot translate.
- `int8` is the default compute type. On Pascal cards (GTX 10-series) `float16`
  runs at 1/64 of `float32`.
- Every watch folder must be bind-mounted into the container.
