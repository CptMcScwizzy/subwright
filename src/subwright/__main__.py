"""Entry point.

Wires configuration, storage, the worker and the web UI together. Deliberately
thin: it constructs things and hands them to code that is tested.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from pathlib import Path

from . import config, selftest
from .db import Database
from .runtime import Runtime
from .transcriber import FasterWhisperTranscriber, TranscriberUnavailable

__version__ = "0.1.0"

log = logging.getLogger("subwright")

# Must be on local disk, never on the NFS watch mount - SQLite over NFS is a
# known source of corruption.
DEFAULT_CONFIG_DIR = Path(os.environ.get("SW_CONFIG_DIR", "/config"))


def setup_logging(level: str) -> None:
    """Log to stdout so `docker logs` sees it.

    The original wrote to a file under the user's home, never rotated it, and
    silently dropped the handler if the path was unwritable.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


def open_database() -> Database | None:
    """Open the settings/history database, or None if the directory is unusable.

    Not fatal: a broken /config should degrade to environment-only settings and
    no history, rather than refusing to transcribe anything.
    """
    try:
        return Database(DEFAULT_CONFIG_DIR / "subwright.db")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not open %s (%s); running without stored settings or history",
                    DEFAULT_CONFIG_DIR, exc)
        return None


def check_gpu(settings: config.Settings) -> int:
    """Load the model on the configured device and report. The only GPU-dependent path."""
    print(f"model        {settings.model}")
    print(f"device       {settings.device}")
    print(f"compute type {settings.compute_type}")
    transcriber = FasterWhisperTranscriber(
        settings.model, device=settings.device, compute_type=settings.compute_type
    )
    try:
        transcriber.load()
    except Exception as exc:  # noqa: BLE001 - report anything, then fail
        print(f"\nFAIL  {type(exc).__name__}: {exc}")
        return 1
    print("\nPASS  model loaded")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Parse once without stored settings so --self-test and --version work even
    # if the database cannot be opened.
    bootstrap, args = config.resolve(argv)

    if args.version:
        print(f"subwright {__version__}")
        return 0

    setup_logging(bootstrap.log_level)

    if args.self_test:
        return selftest.run()

    db = open_database()
    settings = bootstrap
    if db is not None:
        settings, _ = config.resolve(argv, stored=db.load_settings())

    if args.check_gpu:
        return check_gpu(settings)

    log.info("subwright %s starting", __version__)
    for line in settings.describe():
        log.info("  %s", line)

    if settings.compute_type == "float16":
        # Pascal (compute 6.1) has no tensor cores and runs float16 at 1/64 of
        # float32. Worth saying loudly rather than leaving someone to wonder why
        # a job took all night.
        log.warning(
            "compute_type=float16 is very slow on pre-Volta GPUs such as the GTX 10 "
            "series; int8 is normally the right choice"
        )

    if args.demo:
        # Development mode: no GPU, no model, no 3 GB download. Files dropped in
        # ingest/ get placeholder subtitles instantly, so the UI can be worked on
        # end to end. Never reachable without the explicit --demo flag.
        from .selftest import _FakeTranscriber

        log.warning("DEMO MODE - subtitles are placeholder text, not real transcription")
        # Paced and multi-cue so the progress bar and subtitle preview
        # actually do something to look at.
        transcriber = _FakeTranscriber(pace=0.7, count=40)
    else:
        transcriber = FasterWhisperTranscriber(
            settings.model, device=settings.device, compute_type=settings.compute_type
        )

    if db is None:
        # No storage: run the watcher alone. Better than refusing to work.
        from .worker import Worker

        worker = Worker(
            settings.watch_dir, transcriber,
            language=settings.language_or_none,
            poll_interval=settings.poll_interval,
            settle_seconds=settings.settle_seconds,
            keep_backups=settings.keep_backups,
            uid=settings.uid, gid=settings.gid,
        )
        _install_signal_handlers(worker.stop)
        try:
            worker.run_forever()
        except TranscriberUnavailable as exc:
            log.error("%s", exc)
            return 1
        return 0

    runtime = Runtime(settings, db, transcriber)
    runtime.start()
    _install_signal_handlers(runtime.worker.stop)

    if args.no_web:
        log.info("running without the web UI")
        runtime._thread.join()
        return 0

    import uvicorn

    from .web.app import create_app

    app = create_app(
        db,
        settings,
        status_provider=runtime.status_snapshot,
        on_settings_saved=runtime.apply_settings,
        cancel_current=runtime.cancel_current,
        requeue=runtime.requeue,
        reprocess=runtime.reprocess,
        version=__version__,
    )
    log.info("web UI on http://%s:%s", settings.host, settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")

    runtime.stop()
    return 0


def _install_signal_handlers(stop) -> None:
    def handle(signum, _frame):
        log.info("signal %s received; finishing the current job then stopping", signum)
        stop()

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


if __name__ == "__main__":
    raise SystemExit(main())
