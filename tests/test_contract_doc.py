"""docs/CONTRACT.md must stay true.

It maps frozen behaviours to the test names that prove them, and it is the one
document written for someone who does not read Python. A contract that cites
tests which do not exist is worse than no contract: it reads as evidence while
proving nothing.

Six of the fifty names in the first draft were wrong, written from memory. This
is why.
"""

from __future__ import annotations

import re
from pathlib import Path

CONTRACT = Path(__file__).parent.parent / "docs" / "CONTRACT.md"
TESTS = Path(__file__).parent


def _real_test_names() -> set[str]:
    names: set[str] = set()
    for f in TESTS.rglob("test_*.py"):
        names |= set(re.findall(r"^def (test_[a-z0-9_]+)",
                                f.read_text(encoding="utf-8"), re.M))
    return names


def _cited_test_names() -> list[str]:
    return re.findall(r"`(test_[a-z0-9_]+)`", CONTRACT.read_text(encoding="utf-8"))


def test_the_contract_document_exists():
    """srt.py points readers at it."""
    assert CONTRACT.is_file()


def test_every_test_named_in_the_contract_actually_exists():
    missing = sorted(set(_cited_test_names()) - _real_test_names())
    assert not missing, (
        "docs/CONTRACT.md cites tests that do not exist: " + ", ".join(missing)
    )


def test_the_contract_cites_a_meaningful_number_of_tests():
    """Guards against the file being gutted into something that still passes
    the check above by citing nothing at all."""
    assert len(set(_cited_test_names())) >= 40
