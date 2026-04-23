"""Persist 'already seen', 'applied', and 'rejected' post URLs across runs.

Dedup semantics:
  - **Applied / rejected** — permanent (user has made a decision; never re-show)
  - **Seen** — rolling TTL via `config.DEDUP_DAYS`. A delivered post is
     suppressed for the next N days, then re-eligible. This lets long-lived
     job-board listings resurface later without spamming the user daily, and
     is what "no repetition for 3 continuous days" means in the product spec.

The underlying file format is forward/backward-compatible:

    {
      "<key>": {
        "url": "...",
        "first_seen": "<iso-8601-utc>",
        "delivered_at": "<iso-8601-utc>"   # set when recorded post-delivery
      }
    }

`delivered_at` takes precedence over `first_seen` for TTL math. Legacy records
that pre-date the delivered_at field fall back to first_seen so upgrades
don't accidentally re-deliver yesterday's digest.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import config

SEEN_PATH = config.CONFIG_DIR / "seen_posts.json"
APPLIED_PATH = config.CONFIG_DIR / "applied_posts.json"
REJECTED_PATH = config.CONFIG_DIR / "rejected_posts.json"


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def post_key(post: dict[str, Any]) -> str:
    """Stable identity for a post: prefer URN, fall back to URL hash."""
    urn = post.get("post_urn")
    if urn:
        return f"urn:{urn}"
    url = post.get("post_url") or ""
    return "url:" + hashlib.sha1(url.encode("utf-8")).hexdigest()


def _parse_iso(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        # datetime.fromisoformat handles "+00:00" suffix as well as naive ISO
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _active_seen_keys() -> set[str]:
    """Seen entries whose TTL window hasn't elapsed.

    When DEDUP_DAYS <= 0 the TTL is disabled and all seen keys are returned
    (matches the original permanent-dedup behavior).

    Entries without a parseable timestamp are treated as expired — we'd rather
    risk a rare re-delivery than suppress a post forever based on a corrupted
    record.
    """
    data = _load(SEEN_PATH)
    if config.DEDUP_DAYS <= 0:
        return set(data.keys())
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.DEDUP_DAYS)
    out: set[str] = set()
    for k, v in data.items():
        if not isinstance(v, dict):
            continue
        ts = _parse_iso(v.get("delivered_at")) or _parse_iso(v.get("first_seen"))
        if ts is None:
            continue
        if ts >= cutoff:
            out.add(k)
    return out


def _all_known() -> set[str]:
    """Union of TTL-active seen + permanent applied + permanent rejected keys."""
    return (
        _active_seen_keys()
        | set(_load(APPLIED_PATH).keys())
        | set(_load(REJECTED_PATH).keys())
    )


def drop_known(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known = _all_known()
    return [p for p in posts if post_key(p) not in known]


def record_seen(posts: list[dict[str, Any]]) -> None:
    """Stamp posts with first_seen AND delivered_at for TTL-aware dedup.

    Callers should invoke this AFTER successful delivery so the TTL window
    anchors to the moment the user actually received the post in an email.
    Dry-run paths intentionally skip this so state doesn't drift on preview
    runs (pipeline.run_pipeline enforces that).
    """
    seen = _load(SEEN_PATH)
    now = datetime.now(timezone.utc).isoformat()
    for p in posts:
        key = post_key(p)
        existing = seen.get(key) if isinstance(seen.get(key), dict) else {}
        first_seen = existing.get("first_seen") or now if isinstance(existing, dict) else now
        seen[key] = {
            "url": p.get("post_url"),
            "first_seen": first_seen,
            "delivered_at": now,
        }
    _save(SEEN_PATH, seen)


def prune_seen(retain_days: int | None = None) -> int:
    """Remove seen entries older than `retain_days` (default 4× DEDUP_DAYS).

    Without pruning, seen_posts.json grows without bound over months of cron
    runs. We retain entries beyond the TTL window for forensic / debugging
    purposes but drop anything truly stale.

    Returns the number of entries removed so the caller can log it.
    """
    if retain_days is None:
        retain_days = max(config.DEDUP_DAYS * 4, 30)
    data = _load(SEEN_PATH)
    if not data:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=retain_days)
    kept: dict[str, Any] = {}
    removed = 0
    for k, v in data.items():
        if not isinstance(v, dict):
            removed += 1
            continue
        ts = _parse_iso(v.get("delivered_at")) or _parse_iso(v.get("first_seen"))
        if ts is None:
            removed += 1
            continue
        if ts >= cutoff:
            kept[k] = v
        else:
            removed += 1
    if removed:
        _save(SEEN_PATH, kept)
    return removed


def mark_applied(url_or_key: str) -> bool:
    applied = _load(APPLIED_PATH)
    key = _resolve_key(url_or_key)
    if not key:
        return False
    applied[key] = {"url": url_or_key, "at": datetime.now(timezone.utc).isoformat()}
    _save(APPLIED_PATH, applied)
    return True


def mark_rejected(url_or_key: str) -> bool:
    rejected = _load(REJECTED_PATH)
    key = _resolve_key(url_or_key)
    if not key:
        return False
    rejected[key] = {"url": url_or_key, "at": datetime.now(timezone.utc).isoformat()}
    _save(REJECTED_PATH, rejected)
    return True


def _resolve_key(url_or_key: str) -> str:
    """Resolve an input (URL or key string) to the canonical key.

    If input already looks like a key (`urn:...` / `url:...`), pass through.
    Otherwise hash as URL. Also scans `seen_posts.json` for any entry whose
    stored `url` equals this URL — lets `apply <url>` hit a post originally
    recorded under its URN key.
    """
    if url_or_key.startswith(("urn:", "url:")):
        return url_or_key
    # Look in seen file for a urn-keyed entry whose url matches
    for k, v in _load(SEEN_PATH).items():
        if isinstance(v, dict) and v.get("url") == url_or_key:
            return k
    return "url:" + hashlib.sha1(url_or_key.encode("utf-8")).hexdigest()


def stats() -> dict[str, int]:
    return {
        "seen": len(_load(SEEN_PATH)),
        "seen_active": len(_active_seen_keys()),
        "applied": len(_load(APPLIED_PATH)),
        "rejected": len(_load(REJECTED_PATH)),
    }
