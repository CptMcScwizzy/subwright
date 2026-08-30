"""Web UI and its HTTP endpoints.

Server-rendered Jinja2 with HTMX for live updates. No npm, no build step, no
JavaScript bundle - the page is HTML and the browser polls a fragment.

The app is built by create_app() so tests can construct one against a temporary
database and a stub worker.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .. import config
from ..db import Database

log = logging.getLogger(__name__)

TEMPLATES = Path(__file__).parent / "templates"


def _duration(seconds: float | None) -> str:
    if not seconds:
        return "-"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _elapsed_since(iso: str | None) -> str:
    if not iso:
        return "-"
    try:
        started = datetime.fromisoformat(iso)
    except ValueError:
        return "-"
    return _duration((datetime.now() - started).total_seconds())


def create_app(
    db: Database,
    settings: config.Settings,
    *,
    status_provider=None,
    on_settings_saved=None,
    cancel_current=None,
    requeue=None,
    version: str = "0.0.0",
) -> FastAPI:
    """Build the application.

    status_provider() -> dict   live worker status
    on_settings_saved(dict)     called after settings change, so the worker reloads
    cancel_current()            ask the worker to abandon the running job
    requeue(job_row) -> str     put a file back in ingest/; returns a message
    """
    app = FastAPI(title="subwright", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES))
    templates.env.filters["duration"] = _duration
    templates.env.filters["elapsed"] = _elapsed_since

    state: dict[str, Any] = {"settings": settings}

    # Every key the templates touch. The idle fallback must carry all of them:
    # a missing key is Undefined in Jinja, and `Undefined is not none` is
    # true, so an absent "progress" would render a bar rather than hide one.
    IDLE_STATUS = {
        "running": False, "current_file": None, "current_kind": None,
        "started_at": None, "media_duration": 0.0, "last_error": None,
        "processed": 0, "failed": 0,
        "position": 0.0, "cue_count": 0, "last_cue": None, "progress": None,
    }

    def current_status() -> dict:
        if status_provider is None:
            return dict(IDLE_STATUS)
        return status_provider()

    def base_context(request: Request) -> dict:
        return {
            "request": request,
            "version": version,
            "settings": state["settings"],
            "status": current_status(),
        }

    # --- pages ---

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        ctx = base_context(request)
        ctx["jobs"] = db.recent_jobs(limit=10)
        ctx["counts"] = db.counts()
        return templates.TemplateResponse(request, "dashboard.html", ctx)

    @app.get("/status", response_class=HTMLResponse)
    def status_fragment(request: Request):
        """Polled by HTMX every few seconds. Just the live panel."""
        ctx = base_context(request)
        ctx["jobs"] = db.recent_jobs(limit=10)
        ctx["counts"] = db.counts()
        return templates.TemplateResponse(request, "_status.html", ctx)

    @app.get("/history", response_class=HTMLResponse)
    def history(request: Request, status: str | None = None):
        ctx = base_context(request)
        ctx["jobs"] = db.recent_jobs(limit=200, status=status)
        ctx["filter_status"] = status
        return templates.TemplateResponse(request, "history.html", ctx)

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request, saved: int = 0, error: str | None = None):
        ctx = base_context(request)
        ctx["models"] = config.MODELS
        ctx["devices"] = config.DEVICES
        ctx["compute_types"] = config.COMPUTE_TYPES
        ctx["saved"] = bool(saved)
        ctx["error"] = error
        return templates.TemplateResponse(request, "settings.html", ctx)

    @app.post("/settings")
    def save_settings(
        watch_dir: str = Form(...),
        model: str = Form(...),
        language: str = Form(""),
        poll_interval: int = Form(...),
        device: str = Form(...),
        compute_type: str = Form(...),
        settle_seconds: int = Form(...),
        keep_backups: int = Form(...),
        # An unticked checkbox submits nothing at all, so the default here is
        # what 'off' looks like on the wire - it is not a fallback.
        show_preview: bool = Form(False),
    ):
        values = {
            "watch_dir": watch_dir.strip(),
            "model": model,
            "language": language.strip(),
            "poll_interval": poll_interval,
            "device": device,
            "compute_type": compute_type,
            "settle_seconds": settle_seconds,
            "keep_backups": keep_backups,
            "show_preview": show_preview,
        }
        # Validate BEFORE persisting - a bad value must not be able to break the
        # next startup and leave the UI unreachable.
        try:
            candidate = config.Settings(
                **{**{k: getattr(state["settings"], k) for k in config.DEFAULTS}, **values,
                   "watch_dir": Path(values["watch_dir"])}
            )
            candidate.validate()
        except (ValueError, TypeError) as exc:
            return RedirectResponse(f"/settings?error={exc}", status_code=303)

        db.save_settings(values)
        state["settings"] = candidate
        if on_settings_saved:
            on_settings_saved(candidate)
        return RedirectResponse("/settings?saved=1", status_code=303)

    # --- actions ---

    @app.post("/jobs/{job_id}/retry")
    def retry(job_id: int):
        row = db.job(job_id)
        if row is None:
            return RedirectResponse("/history?error=no+such+job", status_code=303)
        if requeue is None:
            return RedirectResponse("/history?error=retry+unavailable", status_code=303)
        try:
            requeue(row)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not raised
            log.exception("retry failed for job %s", job_id)
            return RedirectResponse(f"/history?error={exc}", status_code=303)
        return RedirectResponse("/history?requeued=1", status_code=303)

    @app.post("/cancel")
    def cancel():
        if cancel_current:
            cancel_current()
        return RedirectResponse("/", status_code=303)

    # --- machine-readable ---

    @app.get("/healthz")
    def healthz():
        """Container healthcheck. Deliberately cheap and not GPU-dependent -
        the GPU is shared, and a check that fails because something else is
        using it would be worse than no check."""
        return JSONResponse({"ok": True, "version": version})

    @app.get("/api/status")
    def api_status():
        return JSONResponse(
            {
                "version": version,
                "status": current_status(),
                "counts": db.counts(),
                "settings": {
                    k: str(getattr(state["settings"], k)) for k in config.DEFAULTS
                },
            }
        )

    return app
