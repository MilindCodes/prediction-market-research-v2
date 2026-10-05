"""Repository hygiene checks that must hold before anything is pushed.

The credential test mirrors the `secrets` job in .github/workflows/ci.yml so a
mistake is caught locally, at commit time, rather than after it is public.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Filenames that must never be tracked by git.
FORBIDDEN_NAME = re.compile(
    r"(^|/)(researchproject[0-9]*\.txt|\.env|id_rsa|id_ed25519)$|\.(pem|key|p12|pfx)$",
    re.IGNORECASE,
)

# Credential material that must never appear inside a tracked file.
CREDENTIAL_PATTERNS = [
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"ghp_[A-Za-z0-9]{36}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{22,}"),
    re.compile(rb"sk-[A-Za-z0-9]{32,}"),
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{10,}"),
]


def _tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"git is unavailable: {exc}")
    if out.returncode != 0:  # pragma: no cover
        pytest.skip(f"git ls-files failed: {out.stderr.strip()}")
    return [line for line in out.stdout.splitlines() if line]


class TestNoTrackedCredentials:
    def test_no_credential_filenames_are_tracked(self):
        offenders = [f for f in _tracked_files() if FORBIDDEN_NAME.search(f)]
        assert not offenders, (
            "Credential file(s) are tracked by git: "
            f"{offenders}. Revoke the credential at the provider first, then "
            "remove it from the repository. See SECURITY.md."
        )

    def test_no_credential_material_inside_tracked_files(self):
        offenders = []
        for relative in _tracked_files():
            path = REPO_ROOT / relative
            try:
                blob = path.read_bytes()
            except OSError:
                # Unreadable cloud placeholder; the CI run covers it.
                continue
            if b"\0" in blob[:8192]:
                continue  # binary
            if any(pattern.search(blob) for pattern in CREDENTIAL_PATTERNS):
                offenders.append(relative)
        assert not offenders, (
            f"Credential material found inside tracked file(s): {offenders}. "
            "Revoke it at the provider first. See SECURITY.md."
        )


class TestGovernanceFiles:
    @pytest.mark.parametrize(
        "relative",
        [
            "README.md",
            "LICENSE",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            "SECURITY.md",
            ".gitignore",
            ".gitattributes",
            ".editorconfig",
            "requirements.txt",
            "requirements-dev.txt",
            ".github/PULL_REQUEST_TEMPLATE.md",
            ".github/ISSUE_TEMPLATE/bug_report.md",
            ".github/ISSUE_TEMPLATE/feature_request.md",
            ".github/workflows/ci.yml",
        ],
    )
    def test_file_exists_and_is_not_empty(self, relative):
        path = REPO_ROOT / relative
        assert path.is_file(), f"{relative} is missing"
        assert path.stat().st_size > 0, f"{relative} is empty"


class TestGitignoreCoversSecrets:
    @pytest.mark.parametrize(
        "pattern",
        ["*.pem", "*.key", ".env", "researchproject1.txt", "researchproject2.txt"],
    )
    def test_secret_pattern_is_ignored(self, pattern):
        text = (REPO_ROOT / ".gitignore").read_text()
        assert pattern in text, f".gitignore is missing the '{pattern}' rule"


class TestLineEndings:
    def test_no_tracked_text_file_uses_crlf(self):
        offenders = []
        for relative in _tracked_files():
            if not relative.endswith((".py", ".md", ".yml", ".yaml", ".txt", ".cfg", ".ini")):
                continue
            path = REPO_ROOT / relative
            try:
                blob = path.read_bytes()
            except OSError:
                continue
            if b"\r\n" in blob:
                offenders.append(relative)
        assert not offenders, f"CRLF line endings found in {offenders}. .gitattributes enforces LF."
