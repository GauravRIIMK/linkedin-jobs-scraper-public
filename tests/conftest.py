"""Pytest setup:
- Stub out optional deps (`voyageai`) so tests run on Python 3.14 where voyageai
  can't build. Real production runs on 3.11 where voyageai installs fine.
- Isolate per-test state dirs to a tmp path so tests don't stomp real config/results.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest


def _install_voyageai_stub() -> None:
    if "voyageai" in sys.modules:
        return
    stub = types.ModuleType("voyageai")

    class _FakeClient:
        def __init__(self, *a, **kw): ...
        def embed(self, texts, model=None, input_type=None):
            import numpy as np
            res = types.SimpleNamespace()
            res.embeddings = [np.ones(8, dtype=float).tolist() for _ in texts]
            return res

    stub.Client = _FakeClient
    sys.modules["voyageai"] = stub


_install_voyageai_stub()


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Each test gets its own CONFIG_DIR / RESULTS_DIR and fake API keys."""
    os.environ.setdefault("APIFY_TOKEN", "fake")
    os.environ.setdefault("ANTHROPIC_API_KEY", "fake")
    os.environ.setdefault("VOYAGE_API_KEY", "fake")
    os.environ.setdefault("RESEND_API_KEY", "re_fake")
    os.environ.setdefault("EMAIL_TO", "test@example.com")
    os.environ["DELIVERY_METHOD"] = "resend"

    # Re-import config fresh so env vars stick
    import importlib
    from src import config as cfg

    cfg.CONFIG_DIR = tmp_path / "config"
    cfg.RESULTS_DIR = tmp_path / "results"
    cfg.RESUME_DIR = tmp_path / "resume"
    cfg.PREFS_PATH = cfg.CONFIG_DIR / "role_preferences.json"
    cfg.PROFILE_PATH_TMPL = cfg.CONFIG_DIR / "resume_profile_{track}.json"
    # Priority-ranker tier-1 list AND resume embedding cache also need to
    # point at tmp_path, otherwise tests that write through these paths
    # (e.g. test_priority_ranker._write_tier1) will clobber the real file.
    cfg.INDIAN_TIER1_COMPANIES_PATH = cfg.CONFIG_DIR / "indian_tier1_companies.json"
    cfg.RESUME_EMBEDDING_CACHE = cfg.CONFIG_DIR / "resume_embedding_cache.json"
    cfg.ensure_dirs()

    # Re-import tracking so module-level paths rebind
    from src import tracking
    tracking.SEEN_PATH = cfg.CONFIG_DIR / "seen_posts.json"
    tracking.APPLIED_PATH = cfg.CONFIG_DIR / "applied_posts.json"
    tracking.REJECTED_PATH = cfg.CONFIG_DIR / "rejected_posts.json"

    # priority_ranker caches the tier-1 list on first load; reset that cache
    # between tests so a fresh per-test tier-1 file is actually read.
    try:
        from src import priority_ranker
        priority_ranker.reload_tier1_companies()
    except ImportError:
        pass

    yield
