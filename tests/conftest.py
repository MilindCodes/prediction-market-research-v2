"""Shared pytest fixtures.

The project has no package installation step: modules are imported relative to
the repository root, exactly as the CLI runners do. These fixtures make that
work from pytest without polluting the rest of the session.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Matplotlib must not try to open a window when analysis.plots is imported.
os.environ.setdefault("MPLBACKEND", "Agg")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _preserve_cwd():
    """Restore the working directory after each test.

    ``run_pipeline`` calls ``os.chdir`` at import time so that its relative
    ``data/`` paths resolve. That is fine for a CLI but would leak between
    tests, so it is undone here.
    """
    original = Path.cwd()
    try:
        yield
    finally:
        os.chdir(original)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
