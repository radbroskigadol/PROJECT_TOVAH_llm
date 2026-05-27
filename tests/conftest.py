"""
TOVAH v14 tests/conftest.py — Test isolation.

Every test that boots a kernel touches the on-disk JSON state files
(`tovah_state.json`, `tovah_kernel_ecology.json`, `tovah_packet_log.json`,
…). Without isolation, tests leak state into each other and into the
working directory's persisted files. The autouse fixture below redirects
the working directory to a per-test tmp dir for every test that does not
explicitly opt out by depending on `tmp_path` itself.

S-1 from the audit: most tests in the suite were written without
`tmp_path`/`monkeypatch.chdir(tmp_path)`. This fixture makes that opt-out
the rare case rather than the default.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_cwd(request, tmp_path, monkeypatch):
    """Run every test in a fresh tmp dir so on-disk state files are isolated.

    Tests that explicitly set their own working dir (already using
    `monkeypatch.chdir(...)`) still work — they simply chdir again on top
    of the autouse base.
    """
    monkeypatch.chdir(tmp_path)
    # Ensure required v14 directories exist in the new cwd so kernel boot
    # doesn't fail trying to write into missing folders.
    try:
        from tovah_v14.config.paths import ensure_directories
        ensure_directories()
    except Exception:
        pass
    yield
