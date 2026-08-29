"""Tests for the self-test itself.

A self-test that cannot fail proves nothing, so the important test here is that
breaking the contract makes it report FAIL and exit non-zero.
"""

from subwright import layout, selftest


def test_selftest_passes_on_a_healthy_pipeline(capsys):
    assert selftest.run() == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out
    assert "RESULT" in out


def test_selftest_reports_every_check_it_ran(capsys):
    selftest.run()
    out = capsys.readouterr().out
    # One line per check, plus a blank line and the RESULT line.
    assert out.count("CHECK") >= 15
    assert "PASS" in out


def test_selftest_fails_when_the_success_marker_name_changes(monkeypatch, capsys):
    """Break the Plex/Stash contract and confirm the self-test notices.

    If this test fails, the self-test has stopped being able to detect a broken
    layout, which would make it worse than useless - it would provide false
    reassurance during a migration.
    """
    monkeypatch.setattr(
        layout, "translated_marker", lambda folder: folder / ".not-the-agreed-name"
    )
    assert selftest.run() == 1
    assert "FAIL" in capsys.readouterr().out


def test_selftest_fails_when_subtitles_are_not_written(monkeypatch, capsys):
    monkeypatch.setattr(layout, "srt_for", lambda video: video.with_name("wrong.srt"))
    assert selftest.run() == 1
    assert "FAIL" in capsys.readouterr().out
