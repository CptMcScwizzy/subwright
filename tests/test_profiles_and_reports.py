"""Audio profiles, the diagnostic report, and the history actions.

The profile tests are mostly about direction: which way does each knob move,
and does the more permissive profile actually keep more? Getting a sign wrong
here is invisible in normal use and quietly deletes dialogue, which is exactly
what happened to the setting these replace.
"""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise
from pathlib import Path

import pytest

from subwright import jobs, layout, profiles, report, rules
from subwright.db import Database
from subwright.profiles import DIFFICULT, MAXIMUM, STANDARD, Diagnostics
from subwright.rules import RuleError, WatchRule
from subwright.srt import Cue
from subwright.transcriber import MediaInfo
from subwright.worker import Worker

from .fakes import FakeTranscriber

NOW = datetime(2026, 1, 2, 3, 4, 5)


def _clock():
    return NOW


def _drop(tmp_path: Path, name: str = "Foo.mkv") -> tuple[Path, Path]:
    ingest = tmp_path / "ingest"
    ingest.mkdir(parents=True, exist_ok=True)
    video = ingest / name
    video.write_bytes(b"data")
    return tmp_path, video


# --- the bug these replace ---

def test_the_standard_profile_no_longer_discards_more_than_whisper_would():
    """The inherited script set no_speech_threshold to 0.4, commented "lower:
    less likely to drop quiet passages". A segment is skipped when
    no_speech_prob EXCEEDS this value, so 0.4 discarded MORE than the 0.6
    default, not less. The comment described the opposite of the behaviour."""
    assert STANDARD.no_speech_threshold >= 0.6


def test_each_profile_is_more_permissive_than_the_last():
    """Direction, on every knob at once. A profile called "difficult audio"
    that quietly kept less than Standard would be worse than useless."""
    ladder = [STANDARD, DIFFICULT, MAXIMUM]
    for tighter, looser in pairwise(ladder):
        # Lower VAD threshold: quieter sound still counts as speech.
        assert looser.vad_threshold < tighter.vad_threshold
        # Wider padding: word onsets after a pause are not clipped.
        assert looser.vad_speech_pad_ms > tighter.vad_speech_pad_ms
        # Higher no-speech threshold: harder to declare a segment silent.
        assert looser.no_speech_threshold > tighter.no_speech_threshold
        # Lower log-prob threshold: the low-confidence rescue fires more.
        assert looser.log_prob_threshold < tighter.log_prob_threshold
        # Higher compression ratio: repetitive dialogue is not binned.
        assert looser.compression_ratio_threshold > tighter.compression_ratio_threshold


def test_every_profile_still_translates_into_english():
    for profile in profiles.PROFILES.values():
        assert profile.transcribe_options()["task"] == "translate"


def test_voice_detection_is_never_switched_off_entirely():
    """Without it Whisper hallucinates across long silences, which is a worse
    failure than a gap because it looks like real dialogue."""
    for profile in profiles.PROFILES.values():
        assert profile.transcribe_options()["vad_filter"] is True


def test_an_unknown_profile_falls_back_instead_of_failing():
    """A job must not die because a profile was renamed under it."""
    assert profiles.get("no-such-profile").key == STANDARD.key
    assert profiles.get(None).key == STANDARD.key
    assert profiles.get("").key == STANDARD.key


# --- profiles reach faster-whisper ---

def test_the_folders_profile_is_the_one_actually_used(tmp_path):
    rule = WatchRule("rough", tmp_path / "in", tmp_path / "out", profile="difficult")
    (rule.ingest).mkdir(parents=True)
    (rule.ingest / "Foo.mkv").write_bytes(b"data")

    transcriber = FakeTranscriber()
    Worker([rule], transcriber, clock=_clock, monotonic=lambda: 9e9,
           sleep=lambda _: None).run_once()

    assert transcriber.profiles[0].key == "difficult"


def test_two_folders_can_use_different_profiles(tmp_path):
    a = WatchRule("clean", tmp_path / "a/in", tmp_path / "a/out", profile="standard")
    b = WatchRule("rough", tmp_path / "b/in", tmp_path / "b/out", profile="maximum")
    for rule, name in ((a, "One.mkv"), (b, "Two.mkv")):
        rule.ingest.mkdir(parents=True)
        (rule.ingest / name).write_bytes(b"data")

    transcriber = FakeTranscriber()
    Worker([a, b], transcriber, clock=_clock, monotonic=lambda: 9e9,
           sleep=lambda _: None).run_once()

    used = {path.name: prof.key for (path, _), prof
            in zip(transcriber.calls, transcriber.profiles, strict=True)}
    assert used == {"One.mkv": "standard", "Two.mkv": "maximum"}


def test_a_folder_with_an_unknown_profile_is_rejected_when_saved(tmp_path):
    with pytest.raises(RuleError, match="unknown audio profile"):
        rules.validate_all([
            WatchRule("x", tmp_path / "in", tmp_path / "out", profile="loudest"),
        ])


def test_a_folder_saved_before_profiles_existed_gets_the_standard_one(tmp_path):
    """Rules already stored in deployed databases have no profile key."""
    rule = WatchRule.from_dict({
        "name": "old", "ingest": str(tmp_path / "in"), "output": str(tmp_path / "out"),
        "language": "ja", "enabled": True,
    })
    assert rule.profile == profiles.DEFAULT_PROFILE


# --- diagnostics ---

def test_the_speech_fraction_is_what_survived_voice_detection():
    d = Diagnostics(duration=1000.0, duration_after_vad=150.0)
    assert d.speech_fraction == pytest.approx(0.15)


def test_the_speech_fraction_is_unknown_rather_than_zero_for_an_unknown_duration():
    assert Diagnostics(duration=0.0, duration_after_vad=0.0).speech_fraction is None


def test_confidence_is_averaged_over_the_cues_that_reported_it():
    d = Diagnostics(logprobs=[-0.2, -0.4, -1.2])
    assert d.mean_logprob == pytest.approx(-0.6)
    assert d.doubtful_count == 1


# --- the report file ---

def _render(**kw) -> str:
    defaults = dict(
        video=Path("Foo.mkv"),
        cues=[Cue(0.0, 2.0, "Hello", avg_logprob=-0.2),
              Cue(2.0, 4.0, "Mumbled", avg_logprob=-1.6)],
        diagnostics=Diagnostics(duration=3600.0, duration_after_vad=1800.0,
                                cue_count=2, logprobs=[-0.2, -1.6]),
        profile=STANDARD, model="large-v3", device="cuda", compute_type="int8",
        language="ja", detected_language="ja", language_probability=0.98,
        now=NOW,
    )
    defaults.update(kw)
    return report.render(**defaults)


def test_the_report_says_how_much_audio_was_actually_heard():
    text = _render()
    assert "speech detected" in text
    assert "50%" in text


def test_the_report_says_how_confident_the_model_was():
    text = _render()
    assert "mean confidence" in text
    assert "low confidence" in text


def test_the_report_lists_the_lines_worth_checking():
    text = _render()
    assert "Mumbled" in text, "the doubtful line was not listed"
    assert "Hello" not in text, "a confident line was listed as doubtful"


def test_the_report_records_the_settings_that_produced_it():
    """Without these the file cannot be compared against one from another run,
    which is the whole point of writing it while tuning."""
    text = _render(profile=DIFFICULT)
    assert "Difficult audio" in text
    assert "no-speech threshold" in text
    assert str(DIFFICULT.vad_threshold) in text


def test_the_report_warns_when_voice_detection_ate_the_file():
    text = _render(diagnostics=Diagnostics(duration=3600.0, duration_after_vad=180.0))
    assert "Very little of this file" in text
    assert "Difficult audio" in text


def test_the_report_stays_quiet_when_there_is_nothing_to_warn_about():
    text = _render(
        diagnostics=Diagnostics(duration=3600.0, duration_after_vad=3000.0,
                                logprobs=[-0.1, -0.2]),
        cues=[Cue(0.0, 2.0, "Clear", avg_logprob=-0.1)],
    )
    assert "Very little of this file" not in text
    assert "LEAST CONFIDENT" not in text


def test_a_reused_result_reports_no_confidence_figures():
    """There was no transcription, so inventing numbers for one would be a lie."""
    text = _render(source="sidecar", source_detail="Foo.srt")
    assert "REUSED" in text
    assert "mean confidence" not in text


def test_the_report_is_written_beside_the_subtitles(tmp_path):
    base, video = _drop(tmp_path)
    result = jobs.run_ingest(video, base, FakeTranscriber(), language="ja", now=_clock)
    written = jobs.write_report(
        result, profile=STANDARD, model="large-v3", device="cuda",
        compute_type="int8", language="ja", rule_name="default", now=_clock)

    assert written == base / "Foo" / "Foo.subwright.txt"
    assert written.is_file()
    assert "subwright report for Foo.mkv" in written.read_text(encoding="utf-8")


def test_a_report_that_cannot_be_written_does_not_fail_the_job(tmp_path):
    """A report explains a finished job. Failing to write one must not undo
    the job it is explaining."""
    base, video = _drop(tmp_path)
    result = jobs.run_ingest(video, base, FakeTranscriber(), language="ja", now=_clock)
    # A regular file cannot contain children on any operating system, which
    # makes this a genuine write failure rather than one that depends on
    # permissions the test runner might happen to have.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    result.video = blocker / "Foo.mkv"

    assert jobs.write_report(
        result, profile=STANDARD, model="m", device="cuda", compute_type="int8",
        language="ja", rule_name=None, now=_clock) is None


def test_reports_are_off_unless_asked_for(tmp_path):
    rule = rules.default_for(tmp_path)
    rule.ingest.mkdir(parents=True)
    (rule.ingest / "Foo.mkv").write_bytes(b"data")

    Worker([rule], FakeTranscriber(), clock=_clock, monotonic=lambda: 9e9,
           sleep=lambda _: None).run_once()
    assert not (tmp_path / "Foo" / "Foo.subwright.txt").exists()


def test_reports_are_written_when_switched_on(tmp_path):
    rule = rules.default_for(tmp_path)
    rule.ingest.mkdir(parents=True)
    (rule.ingest / "Foo.mkv").write_bytes(b"data")

    Worker([rule], FakeTranscriber(), clock=_clock, monotonic=lambda: 9e9,
           sleep=lambda _: None, write_reports=True).run_once()
    assert (tmp_path / "Foo" / "Foo.subwright.txt").is_file()


# --- history actions ---

def test_a_single_entry_can_be_removed(tmp_path):
    db = Database(tmp_path / "h.db")
    keep = db.start_job("ingest", Path("/x/keep.mkv"))
    drop = db.start_job("ingest", Path("/x/drop.mkv"))
    db.finish_job(keep)
    db.finish_job(drop)

    assert db.delete_job(drop) is True
    assert [r["id"] for r in db.recent_jobs()] == [keep]
    assert db.delete_job(drop) is False


def test_the_history_can_be_cleared(tmp_path):
    db = Database(tmp_path / "h.db")
    for name in ("a.mkv", "b.mkv"):
        db.finish_job(db.start_job("ingest", Path(f"/x/{name}")))

    assert db.clear_jobs() == 2
    assert db.recent_jobs() == []


def test_clearing_the_history_keeps_a_job_that_is_still_running(tmp_path):
    """Deleting the row for work happening right now would leave the dashboard
    describing a job it can no longer find."""
    db = Database(tmp_path / "h.db")
    db.finish_job(db.start_job("ingest", Path("/x/done.mkv")))
    running = db.start_job("ingest", Path("/x/busy.mkv"))

    assert db.clear_jobs() == 1
    assert [r["id"] for r in db.recent_jobs()] == [running]


def test_a_redo_regenerates_in_place_rather_than_ingesting_again(tmp_path):
    """Retry sends a file back through ingest, which lands it in a NEW output
    folder. Redo rewrites the subtitles where they already are, so the two
    results are directly comparable."""
    rule = rules.default_for(tmp_path)
    rule.ingest.mkdir(parents=True)
    (rule.ingest / "Foo.mkv").write_bytes(b"data")

    worker = Worker([rule], FakeTranscriber(), clock=_clock, monotonic=lambda: 9e9,
                    sleep=lambda _: None)
    worker.run_once()

    finished = tmp_path / "Foo" / "Foo.mkv"
    first = layout.srt_for(finished).read_text(encoding="utf-8")

    worker.request_redo(finished)
    assert worker.run_once() == 1

    # Same folder, no second copy of the video anywhere.
    assert not (tmp_path / "Foo_20260102_030405").exists()
    assert finished.is_file()
    assert layout.srt_for(finished).read_text(encoding="utf-8") == first
    # And the previous subtitles were kept.
    assert list((tmp_path / "Foo").glob("Foo.srt.*.bak")), "no backup of the old subtitles"


def test_a_redo_uses_the_profile_of_the_folder_the_file_lives_in(tmp_path):
    rule = WatchRule("rough", tmp_path / "in", tmp_path / "out", profile="maximum")
    rule.ingest.mkdir(parents=True)
    (rule.ingest / "Foo.mkv").write_bytes(b"data")

    transcriber = FakeTranscriber()
    worker = Worker([rule], transcriber, clock=_clock, monotonic=lambda: 9e9,
                    sleep=lambda _: None)
    worker.run_once()

    worker.request_redo(tmp_path / "out" / "Foo" / "Foo.mkv")
    worker.run_once()

    assert transcriber.profiles[-1].key == "maximum"


def test_asking_for_the_same_redo_twice_only_runs_it_once(tmp_path):
    rule = rules.default_for(tmp_path)
    worker = Worker([rule], FakeTranscriber(), clock=_clock, monotonic=lambda: 9e9,
                    sleep=lambda _: None)
    video = tmp_path / "Foo" / "Foo.mkv"
    worker.request_redo(video)
    worker.request_redo(video)
    assert worker.pending_redo == [video]


def test_media_info_reports_what_survived_voice_detection():
    info = MediaInfo(duration=100.0, duration_after_vad=20.0)
    assert info.duration_after_vad == 20.0


def test_the_report_does_not_recommend_the_profile_already_in_use(tmp_path):
    """Telling someone on 'Difficult audio' to try 'Difficult audio' is worse
    than saying nothing - it reads like the tool is not paying attention."""
    text = _render(profile=DIFFICULT,
                   diagnostics=Diagnostics(duration=3600.0, duration_after_vad=180.0))
    assert "try the 'Maximum recall' profile" in text
    assert "try the 'Difficult audio' profile" not in text


def test_the_report_admits_when_no_further_profile_would_help(tmp_path):
    text = _render(profile=MAXIMUM,
                   diagnostics=Diagnostics(duration=3600.0, duration_after_vad=180.0))
    assert "most permissive profile" in text
    assert "try the" not in text


def test_the_old_subtitles_stay_readable_while_a_redo_is_running(tmp_path):
    """Found on hardware: a redo killed mid-transcribe left the video with a
    .bak and no .srt at all, because the original was MOVED aside before the
    work started and only restored by an exception handler a kill never
    reaches. Copying instead means there is no window with no subtitles."""
    folder = tmp_path / "Foo"
    folder.mkdir(parents=True)
    video = folder / "Foo.mkv"
    video.write_bytes(b"data")
    layout.srt_for(video).write_text("the original subtitles", encoding="utf-8")

    seen: dict = {}

    class WatchingTranscriber(FakeTranscriber):
        def transcribe(self, path, language, profile=None):
            # Mid-job: this is the window a kill would land in.
            seen["srt_present"] = layout.srt_for(path).exists()
            seen["srt_text"] = layout.srt_for(path).read_text(encoding="utf-8")
            return super().transcribe(path, language, profile)

    jobs.run_reprocess(video, folder, WatchingTranscriber(), language="ja", now=_clock)

    assert seen["srt_present"], "the video had no subtitles while the redo ran"
    assert seen["srt_text"] == "the original subtitles"


def test_a_failed_redo_leaves_the_original_subtitles_and_no_stray_backup(tmp_path):
    folder = tmp_path / "Foo"
    folder.mkdir(parents=True)
    video = folder / "Foo.mkv"
    video.write_bytes(b"data")
    layout.srt_for(video).write_text("the original subtitles", encoding="utf-8")

    with pytest.raises(RuntimeError):
        jobs.run_reprocess(video, folder,
                           FakeTranscriber(raise_on_call=RuntimeError("no audio")),
                           language="ja", now=_clock)

    assert layout.srt_for(video).read_text(encoding="utf-8") == "the original subtitles"
    assert not list(folder.glob("*.bak")), "a pointless backup was left behind"
