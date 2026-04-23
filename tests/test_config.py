"""Config: env loading, dir creation, secret assertions, daily-run defaults."""
from __future__ import annotations

import os

import pytest

from src import config


def test_ensure_dirs_creates_missing_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESUME_DIR", tmp_path / "resume")
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    config.ensure_dirs()
    assert (tmp_path / "resume").is_dir()
    assert (tmp_path / "config").is_dir()
    assert (tmp_path / "results").is_dir()


def test_assert_secrets_raises_on_missing(monkeypatch):
    monkeypatch.delenv("DOES_NOT_EXIST", raising=False)
    with pytest.raises(RuntimeError, match="DOES_NOT_EXIST"):
        config.assert_secrets("DOES_NOT_EXIST")


def test_assert_secrets_passes_when_present(monkeypatch):
    monkeypatch.setenv("EXISTS", "yes")
    config.assert_secrets("EXISTS")  # no raise


def test_delivery_method_defaults_to_resend():
    assert config.DELIVERY_METHOD in ("resend", "gmail")


def test_posted_limit_days_from_env(monkeypatch):
    # Reload config with overridden env
    monkeypatch.setenv("POSTED_LIMIT_DAYS", "5")
    import importlib
    import src.config as c
    importlib.reload(c)
    assert c.POSTED_LIMIT_DAYS == 5


def test_gform_direct_regexes_non_empty():
    assert len(config.GFORM_DIRECT_REGEXES) >= 3


def test_shortener_hosts_include_common():
    assert "lnkd.in" in config.SHORTENER_HOSTS
    assert "bit.ly" in config.SHORTENER_HOSTS
    assert "tinyurl.com" in config.SHORTENER_HOSTS


def test_claude_models_configured():
    assert config.CLAUDE_MODEL_HEAVY
    assert config.CLAUDE_MODEL_LIGHT
    assert config.VOYAGE_MODEL
