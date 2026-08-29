"""Tests for settings resolution and the frozen defaults."""

from pathlib import Path

import pytest

from subwright import config


def test_defaults_match_the_original_script():
    s, _ = config.resolve([], env={})
    assert s.watch_dir == Path("/mnt/data/translate")
    assert s.model == "large-v3"
    assert s.language_or_none is None  # auto-detect
    assert s.poll_interval == 30
    assert s.settle_seconds == 10


def test_gpu_defaults_suit_a_pascal_card():
    s, _ = config.resolve([], env={})
    assert s.device == "cuda"  # fail loudly rather than fall back to CPU
    assert s.compute_type == "int8"  # float16 is 1/64 speed on compute 6.1


def test_environment_overrides_defaults():
    s, _ = config.resolve([], env={"SW_MODEL": "medium", "SW_POLL_INTERVAL": "5"})
    assert s.model == "medium"
    assert s.poll_interval == 5


def test_cli_flag_overrides_environment():
    s, _ = config.resolve(["--model", "small"], env={"SW_MODEL": "medium"})
    assert s.model == "small"


def test_stored_settings_override_environment():
    s, _ = config.resolve([], env={"SW_MODEL": "medium"}, stored={"model": "base"})
    assert s.model == "base"


def test_cli_flag_beats_stored_settings():
    s, _ = config.resolve(["--model", "small"], env={}, stored={"model": "base"})
    assert s.model == "small"


def test_original_short_flags_still_work():
    s, _ = config.resolve(["-w", "/data", "-m", "medium", "-l", "ja", "-p", "15"], env={})
    assert s.watch_dir == Path("/data")
    assert s.model == "medium"
    assert s.language_or_none == "ja"
    assert s.poll_interval == 15


def test_blank_environment_variable_is_ignored():
    s, _ = config.resolve([], env={"SW_MODEL": ""})
    assert s.model == "large-v3"


def test_turbo_model_is_not_offered_because_it_cannot_translate():
    assert not any("turbo" in m for m in config.MODELS)


def test_unknown_model_on_the_command_line_is_rejected():
    with pytest.raises(SystemExit):  # argparse rejects it at parse time
        config.resolve(["--model", "nonsense"], env={})


def test_unknown_stored_model_is_rejected():
    with pytest.raises(ValueError):
        config.resolve([], env={}, stored={"model": "nonsense"})


def test_zero_poll_interval_is_rejected():
    with pytest.raises(ValueError):
        config.resolve(["--poll-interval", "0"], env={})


def test_non_numeric_environment_value_names_the_variable():
    with pytest.raises(ValueError, match="SW_POLL_INTERVAL"):
        config.resolve([], env={"SW_POLL_INTERVAL": "soon"})


def test_describe_lists_every_effective_value():
    s, _ = config.resolve([], env={})
    described = "\n".join(s.describe())
    assert "model = large-v3" in described
    assert "compute_type = int8" in described
