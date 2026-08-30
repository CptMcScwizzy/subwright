"""Settings.

Precedence: CLI flag > stored settings > environment > default.

Environment seeds the first run; after that the UI writes to the database and
becomes the source of truth. A CLI flag always wins, so a container can be run
once with an override without disturbing what is stored.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULTS = {
    "watch_dir": "/mnt/data/translate",
    "model": "large-v3",
    "language": "",  # empty means auto-detect
    "poll_interval": 30,
    "device": "cuda",
    "compute_type": "int8",
    "settle_seconds": 10,
    "keep_backups": 3,
    "uid": 1000,
    "gid": 1000,
    "log_level": "INFO",
    "host": "0.0.0.0",
    "port": 8420,
    # Whether the dashboard shows the line of subtitle currently being
    # produced. On by default because watching it is how you tell at a
    # glance that a job is really working rather than merely running.
    "show_preview": True,
}

# large-v3-turbo is deliberately absent: it cannot translate, only transcribe,
# and translation is the entire point of this application.
MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
DEVICES = ["cuda", "cpu", "auto"]
COMPUTE_TYPES = ["int8", "int8_float16", "float16", "float32"]

ENV_PREFIX = "SW_"


@dataclass
class Settings:
    watch_dir: Path = Path(DEFAULTS["watch_dir"])
    model: str = DEFAULTS["model"]
    language: str = DEFAULTS["language"]
    poll_interval: int = DEFAULTS["poll_interval"]
    device: str = DEFAULTS["device"]
    compute_type: str = DEFAULTS["compute_type"]
    settle_seconds: int = DEFAULTS["settle_seconds"]
    keep_backups: int = DEFAULTS["keep_backups"]
    uid: int = DEFAULTS["uid"]
    gid: int = DEFAULTS["gid"]
    log_level: str = DEFAULTS["log_level"]
    host: str = DEFAULTS["host"]
    port: int = DEFAULTS["port"]
    show_preview: bool = DEFAULTS["show_preview"]

    @property
    def language_or_none(self) -> str | None:
        """Whisper wants None for auto-detect; a blank string is easier in a form."""
        return self.language or None

    def validate(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}; choose from {MODELS}")
        if self.device not in DEVICES:
            raise ValueError(f"unknown device {self.device!r}; choose from {DEVICES}")
        if self.compute_type not in COMPUTE_TYPES:
            raise ValueError(
                f"unknown compute_type {self.compute_type!r}; choose from {COMPUTE_TYPES}"
            )
        if self.poll_interval < 1:
            raise ValueError("poll_interval must be at least 1 second")
        if self.settle_seconds < 0:
            raise ValueError("settle_seconds must not be negative")
        if self.keep_backups < 0:
            raise ValueError("keep_backups must not be negative")

    def describe(self) -> list[str]:
        """Every effective value, logged at startup so the log says what is running."""
        return [f"{f.name} = {getattr(self, f.name)}" for f in fields(self)]


TRUTHY = {"1", "true", "yes", "on"}
FALSEY = {"0", "false", "no", "off"}


def _coerce(name: str, raw: str):
    default = DEFAULTS[name]
    # bool before int: bool IS an int in Python, and int("true") raises.
    if isinstance(default, bool):
        low = raw.strip().lower()
        if low in TRUTHY:
            return True
        if low in FALSEY:
            return False
        raise ValueError(f"expected true or false, got {raw!r}")
    if isinstance(default, int):
        return int(raw)
    return raw


def from_env(env: dict | None = None) -> dict:
    """Read SW_* variables. Unset or blank variables are ignored."""
    env = os.environ if env is None else env
    out = {}
    for name in DEFAULTS:
        raw = env.get(f"{ENV_PREFIX}{name.upper()}")
        if raw is None or raw == "":
            continue
        try:
            out[name] = _coerce(name, raw)
        except ValueError as exc:
            raise ValueError(
                f"{ENV_PREFIX}{name.upper()}={raw!r} is not valid: {exc}"
            ) from exc
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="subwright",
        description="Watch a folder and generate English subtitles for video using Whisper.",
    )
    # Short flags kept identical to the original script.
    p.add_argument("-w", "--watch-dir")
    p.add_argument("-m", "--model", choices=MODELS)
    p.add_argument("-l", "--language")
    p.add_argument("-p", "--poll-interval", type=int)
    p.add_argument("--device", choices=DEVICES)
    p.add_argument("--compute-type", choices=COMPUTE_TYPES)
    p.add_argument("--settle-seconds", type=int)
    p.add_argument("--keep-backups", type=int)
    p.add_argument("--uid", type=int)
    p.add_argument("--gid", type=int)
    p.add_argument("--log-level")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument(
        "--self-test",
        action="store_true",
        help="run the pipeline against a temporary tree and report PASS/FAIL",
    )
    p.add_argument(
        "--check-gpu",
        action="store_true",
        help="load the model on the configured device and report",
    )
    p.add_argument(
        "--no-web", action="store_true", help="run the watcher without the web UI"
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="use a fake transcriber - no GPU, no model download. For working on "
             "the UI locally: dropped files are 'transcribed' instantly with "
             "placeholder subtitles.",
    )
    p.add_argument("--version", action="store_true")
    return p


def resolve(
    argv: list[str] | None = None,
    *,
    env: dict | None = None,
    stored: dict | None = None,
) -> tuple[Settings, argparse.Namespace]:
    """Merge defaults, environment, stored settings and CLI flags."""
    args = build_parser().parse_args(argv)

    values = dict(DEFAULTS)
    values.update(from_env(env))
    if stored:
        values.update({k: v for k, v in stored.items() if k in DEFAULTS})
    for name in DEFAULTS:
        supplied = getattr(args, name, None)
        if supplied is not None:
            values[name] = supplied

    values["watch_dir"] = Path(values["watch_dir"])
    settings = Settings(**values)
    settings.validate()
    return settings, args
