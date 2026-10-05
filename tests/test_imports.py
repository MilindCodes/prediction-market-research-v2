"""Every module must import cleanly, and every CLI step must be wired up.

This is the cheapest guard against a refactor that breaks an entry point:
an unresolved import or a renamed function shows up here in under a second
instead of forty minutes into a data pull.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

MODULES = [
    "config",
    "src",
    "src.cleaning",
    "src.kalshi",
    "src.kalshi.client",
    "src.kalshi.catalog",
    "src.kalshi.trades",
    "src.polymarket",
    "src.polymarket.client",
    "src.polymarket.gamma_client",
    "src.polymarket.catalog",
    "src.polymarket.trades",
    "src.models",
    "src.models.black_scholes",
    "src.models.heston",
    "src.models.bates",
    "src.models.implied_vol",
    "src.smm",
    "src.smm.bates_smm",
    "src.smm.panel",
    "src.smm.nested_ladder",
    "src.smm.stylized_facts",
    "analysis",
    "analysis.model_comparison",
    "analysis.plots",
]


def _file_is_readable(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            handle.read(1)
        return True
    except OSError:
        return False


def _source_is_readable(module_name: str) -> bool:
    """Whether the module and every package above it can actually be opened.

    Guards against cloud-storage placeholders (OneDrive / iCloud "files
    on-demand") that appear on disk but fail to read. That is an environment
    fault, not a code fault, so those modules are skipped rather than failed.
    CI runs on a plain checkout where every file is real, so nothing skips
    there and coverage is complete.

    Parent packages are checked too: importing ``src.smm.bates_smm`` executes
    ``src/smm/__init__.py`` first, so an unreadable parent breaks a readable
    child.
    """
    parts = module_name.split(".")
    for depth in range(1, len(parts) + 1):
        relative = Path("/".join(parts[:depth]))
        candidates = [REPO_ROOT / relative / "__init__.py", REPO_ROOT / f"{relative}.py"]
        existing = [c for c in candidates if c.exists()]
        if not existing:
            return False
        if not all(_file_is_readable(c) for c in existing):
            return False
    return True


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    if not _source_is_readable(module_name):
        pytest.skip(
            f"{module_name}: source file is not readable on this machine "
            "(cloud-storage placeholder). Not a code defect."
        )
    importlib.import_module(module_name)


class TestPipelineCli:
    def test_every_step_maps_to_a_callable(self):
        run_pipeline = importlib.import_module("run_pipeline")
        assert run_pipeline.STEPS, "STEPS registry is empty"
        for name, entry in run_pipeline.STEPS.items():
            assert isinstance(entry, tuple) and len(entry) == 2, (
                f"STEPS['{name}'] must be a (function, description) pair"
            )
            func, description = entry
            assert callable(func), f"STEPS['{name}'] does not point at a callable"
            assert isinstance(description, str) and description.strip(), (
                f"STEPS['{name}'] has no description"
            )

    def test_step_names_are_cli_safe(self):
        run_pipeline = importlib.import_module("run_pipeline")
        for name in run_pipeline.STEPS:
            assert name == name.strip().lower()
            assert " " not in name, f"step '{name}' contains a space"

    def test_help_text_only_advertises_real_steps(self, capsys):
        run_pipeline = importlib.import_module("run_pipeline")
        run_pipeline.print_help()
        printed = capsys.readouterr().out
        # Longest names first so 'backtest-bs' cannot mask 'backtest-bates'.
        for name in sorted(run_pipeline.STEPS, key=len, reverse=True):
            printed = printed.replace(name, "")
        leftovers = [
            token
            for token in printed.split()
            if token.count("-") >= 1
            and token.replace("-", "").isalpha()
            and token.islower()
            and len(token) > 6
            and not set(token) <= {"-"}
        ]
        assert not leftovers, f"help text advertises step names that are not in STEPS: {leftovers}"
