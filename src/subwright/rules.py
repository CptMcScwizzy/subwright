"""Watch rules: which folders are watched, and where their output goes.

Before this existed there was one watch folder, and `ingest/`, `reprocess/` and
the output folders were all fixed positions inside it. That is still exactly
what you get by default — `default_for()` builds it — but it is now one rule
among possibly several rather than the only possible shape.

A rule is the unit the worker iterates. Each carries its own source language,
because the useful case is a Japanese folder and a Korean folder side by side;
a single global language forces a guess on material you already know.

Output paths are still constructed by `layout`, which remains the only module
that decides what a folder is called. A rule says *where*, layout says *what*.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import languages, layout


class RuleError(ValueError):
    """A rule set that would misbehave if it were allowed to run."""


@dataclass(frozen=True)
class WatchRule:
    name: str
    ingest: Path
    output: Path
    # Optional: regenerating subtitles in place is a manual, occasional thing,
    # and a rule that only ever receives new files does not need one.
    reprocess: Path | None = None
    language: str = ""  # "" means auto-detect, as everywhere else
    enabled: bool = True

    @property
    def language_or_none(self) -> str | None:
        return self.language or None

    @property
    def excluded_dirs(self) -> set[Path]:
        """Folders under output that are inputs, not results.

        The resume scan walks output looking for interrupted jobs. When ingest
        or reprocess sit inside output - which is the default layout - they must
        not be mistaken for one.
        """
        out = {self.ingest}
        if self.reprocess is not None:
            out.add(self.reprocess)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ingest": str(self.ingest),
            "output": str(self.output),
            "reprocess": str(self.reprocess) if self.reprocess else "",
            "language": self.language,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WatchRule:
        reprocess = (raw.get("reprocess") or "").strip()
        return cls(
            name=(raw.get("name") or "").strip() or "unnamed",
            ingest=Path(str(raw["ingest"]).strip()),
            output=Path(str(raw["output"]).strip()),
            reprocess=Path(reprocess) if reprocess else None,
            language=(raw.get("language") or "").strip(),
            enabled=bool(raw.get("enabled", True)),
        )

    def validate(self) -> None:
        if not self.name.strip():
            raise RuleError("a rule needs a name")
        if not languages.is_valid(self.language):
            raise RuleError(
                f"{self.name}: unknown language {self.language!r}; use a code "
                f"such as ja, or leave it blank to auto-detect"
            )
        if self.ingest == self.output:
            # Videos are moved into <output>/<stem>/, so this would not corrupt
            # anything, but a finished folder sitting in the drop folder is
            # confusing enough to be worth refusing.
            raise RuleError(f"{self.name}: ingest and output must be different folders")
        if self.reprocess is not None and self.reprocess == self.ingest:
            raise RuleError(f"{self.name}: reprocess and ingest must be different folders")


def default_for(base: Path, *, language: str = "", name: str = "default") -> WatchRule:
    """The original single-folder layout, expressed as a rule.

    This is what every existing installation gets when it starts up with no
    rules configured, so upgrading changes nothing on disk.
    """
    return WatchRule(
        name=name,
        ingest=layout.ingest_dir(base),
        output=base,
        reprocess=layout.reprocess_dir(base),
        language=language,
    )


def validate_all(rules: list[WatchRule]) -> None:
    """Check a whole rule set, not just each rule alone.

    The interesting failures are between rules rather than within one.
    """
    if not rules:
        raise RuleError("at least one watch rule is required")

    for rule in rules:
        rule.validate()

    names = [r.name.strip().lower() for r in rules]
    duplicate_name = next((n for n in names if names.count(n) > 1), None)
    if duplicate_name:
        raise RuleError(f"two rules are both called {duplicate_name!r}")

    # Two rules watching one folder would both claim the same file, and the
    # loser would fail on a video that had already been moved away.
    ingests = [r.ingest for r in rules if r.enabled]
    duplicate_ingest = next((i for i in ingests if ingests.count(i) > 1), None)
    if duplicate_ingest:
        raise RuleError(f"two rules both watch {duplicate_ingest}")

    # A rule whose ingest sits inside another rule's output would have its
    # dropped files seen as interrupted jobs by that rule's resume scan.
    for rule in rules:
        for other in rules:
            if rule is other or not other.enabled:
                continue
            if rule.ingest.parent == other.output and rule.ingest not in other.excluded_dirs:
                raise RuleError(
                    f"{rule.name}: its ingest folder is directly inside "
                    f"{other.name}'s output folder, which would confuse resume"
                )


def rename(rule: WatchRule, name: str) -> WatchRule:
    return replace(rule, name=name)
