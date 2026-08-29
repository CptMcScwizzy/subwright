"""Entry point.

Wires configuration to the worker, and dispatches the diagnostic subcommands.
Deliberately thin - it constructs things and hands them to code that is tested.
"""

from __future__ import annotations

import logging
import signal
import sys

from . import config, selftest
from .transcriber import FasterWhisperTranscriber, TranscriberUnavailable
from .worker import Worker

__version__ = "0.1.0"

log = logging.getLogger("subwright")


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
    except TranscriberUnavailable as exc:
        print(f"\nFAIL  {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - report anything, then fail
        print(f"\nFAIL  {type(exc).__name__}: {exc}")
        return 1
    print("\nPASS  model loaded")
    return 0


def main(argv: list[str] | None = None) -> int:
    settings, args = config.resolve(argv)

    if args.version:
        print(f"subwright {__version__}")
        return 0

    setup_logging(settings.log_level)

    if args.self_test:
        return selftest.run()

    if args.check_gpu:
        return check_gpu(settings)

    log.info("subwright %s starting", __version__)
    for line in settings.describe():
        log.info("  %s", line)

    if settings.compute_type == "float16":
        # Pascal (compute 6.1) has no tensor cores and runs float16 at 1/64 of
        # float32. Worth saying loudly rather than leaving someone to wonder why
        # a job takes all night.
        log.warning(
            "compute_type=float16 is very slow on pre-Volta GPUs such as the GTX 10 series; "
            "int8 is normally the right choice"
        )

    transcriber = FasterWhisperTranscriber(
        settings.model, device=settings.device, compute_type=settings.compute_type
    )

    worker = Worker(
        settings.watch_dir,
        transcriber,
        language=settings.language_or_none,
        poll_interval=settings.poll_interval,
        settle_seconds=settings.settle_seconds,
        keep_backups=settings.keep_backups,
        uid=settings.uid,
        gid=settings.gid,
    )

    def handle_signal(signum, _frame):
        log.info("signal %s received; finishing the current job then stopping", signum)
        worker.stop()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        worker.run_forever()
    except TranscriberUnavailable as exc:
        # Fatal on purpose. The original fell back to CPU and ran 20-50x slower
        # with one warning line, which looks identical to "the GPU is busy".
        log.error("%s", exc)
        log.error("set device=cpu explicitly if that is what you want")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
