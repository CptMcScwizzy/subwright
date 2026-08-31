# The original

`translate_watcher.py` is the script subwright replaces, copied verbatim from
the machine it ran on. It is kept here unmodified as the reference for what the
behaviour used to be, and is never edited.

`translate-watcher.service` is the systemd unit that ran it. It is verbatim
apart from the account name, which was replaced with `mediauser` — the original
is of no interest to anyone and the path it appears in is host-specific
regardless.

Two things about it are worth recording.

**The committed copy had drifted.** The version in the original repository was
53 lines behind the one actually running. A `remux_if_needed()` function had
been added on the host to repair downloads with junk prepended to the container
header — files where `ffprobe` finds no audio stream and faster-whisper then
raises `tuple index out of range`. It had fired twice and succeeded both times.
That fix existed in exactly one place, on a machine with no backups.

**Bugs carried across, and what happened to them.**

| Original behaviour | Now |
|---|---|
| `.srt` streamed straight into the final path; a crash left a truncated file that looked finished, with the video already moved out of `ingest/` and nothing that would ever look at it again | Written to a scratch file and renamed atomically; a `.processing` claim makes the job resumable |
| `remux_if_needed()` did `unlink()` then `rename()`, so the only copy of the video briefly did not exist | Renamed into place; the original is never absent |
| `except (TimeoutExpired, Exception)` swallowed everything | Explicit exception types |
| CUDA failure fell back to CPU at 20–50× slower, with one warning line | `device=cuda` fails loudly; `auto` is opt-in |
| `.srt.bak` accumulated forever | Newest N kept |
| No SIGTERM handling | Graceful drain |
| Subtitle numbering skipped indices for dropped empty segments | Contiguous |
| ETA extrapolated from media duration, so VAD made it swing from 8h to 40m | Removed; speed relative to realtime is shown instead |
| Log written to `$HOME`, never rotated, silently dropped if unwritable | stdout, plus job history in the UI |

The hand-tuned Whisper parameters — the VAD thresholds, `beam_size`,
`no_speech_threshold`, `compression_ratio_threshold` — were carried across
unchanged and are constants rather than settings. They were tuned against real
material; making them adjustable would add options nobody can evaluate and
several untested code paths.
