#!/usr/bin/env python3
"""Convenience launcher for local development.

Right-click this file in PyCharm and choose Run, or from a terminal:

    python run.py

It starts subwright in DEMO MODE by default: a fake transcriber, so there is no
GPU requirement and no 3 GB model download. Files dropped into the watch folder
get placeholder subtitles instantly, which is what you want when working on the
UI. The web interface is at http://localhost:8420.

Anything you pass through is forwarded, and passing any flag turns demo mode off
unless you also pass --demo:

    python run.py --device cpu            # real transcription on the CPU (slow)
    python run.py --demo -p 5             # demo mode, poll every 5 seconds

For production this file is not used at all - the container runs
`python -m subwright` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
# src layout: make the package importable without needing an editable install.
sys.path.insert(0, str(ROOT / "src"))

# Somewhere harmless to watch, created if missing, ignored by git.
DEV_WATCH = ROOT / "dev-watch"
DEV_CONFIG = ROOT / "dev-config"


def main() -> int:
    import os

    args = sys.argv[1:]

    # With no arguments at all, pick sensible development defaults.
    if not args:
        (DEV_WATCH / "ingest").mkdir(parents=True, exist_ok=True)
        (DEV_WATCH / "reprocess").mkdir(parents=True, exist_ok=True)
        DEV_CONFIG.mkdir(parents=True, exist_ok=True)
        args = ["--demo", "-w", str(DEV_WATCH), "-p", "5"]
        os.environ.setdefault("SW_CONFIG_DIR", str(DEV_CONFIG))
        print(f"Watch folder : {DEV_WATCH}")
        print(f"Drop videos  : {DEV_WATCH / 'ingest'}")
        print("Web UI       : http://localhost:8420")
        print("Mode         : DEMO (placeholder subtitles, no GPU, no model download)")
        print()
    else:
        os.environ.setdefault("SW_CONFIG_DIR", str(DEV_CONFIG))
        DEV_CONFIG.mkdir(parents=True, exist_ok=True)

    from subwright.__main__ import main as real_main

    return real_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
