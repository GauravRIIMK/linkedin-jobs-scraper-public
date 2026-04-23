"""Tracking: seen / applied / rejected dedup + URN-aware lookups."""
from __future__ import annotations

import json

from src import tracking


def test_post_key_prefers_urn():
    post = {"post_urn": "urn:li:activity:123", "post_url": "https://lnkd.in/x"}
    assert tracking.post_key(post) == "urn:urn:li:activity:123"


def test_post_key_falls_back_to_url_hash():
    post = {"post_url": "https://linkedin.com/feed/update/urn:li:activity:999/"}
    key = tracking.post_key(post)
    assert key.startswith("url:")
    assert len(key) > 10


def test_post_key_empty_post():
    key = tracking.post_key({})
    assert key.startswith("url:")


def test_drop_known_removes_seen():
    posts = [
        {"post_urn": "A", "post_url": "https://linkedin.com/a"},
        {"post_urn": "B", "post_url": "https://linkedin.com/b"},
    ]
    tracking.record_seen([posts[0]])
    out = tracking.drop_known(posts)
    assert len(out) == 1
    assert out[0]["post_urn"] == "B"


def test_drop_known_empty_state():
    posts = [{"post_urn": "X", "post_url": "https://linkedin.com/x"}]
    assert len(tracking.drop_known(posts)) == 1


def test_mark_applied_persists():
    url = "https://linkedin.com/feed/update/urn:li:activity:555/"
    assert tracking.mark_applied(url)
    data = json.loads(tracking.APPLIED_PATH.read_text())
    assert any(v.get("url") == url for v in data.values())


def test_mark_rejected_persists():
    url = "https://linkedin.com/feed/update/urn:li:activity:777/"
    assert tracking.mark_rejected(url)
    data = json.loads(tracking.REJECTED_PATH.read_text())
    assert any(v.get("url") == url for v in data.values())


def test_apply_with_url_finds_seen_urn_entry():
    """If a post was recorded under its URN key, `apply <url>` should still mark
    the same logical post rather than creating a duplicate url-hash entry."""
    post = {
        "post_urn": "urn:li:activity:999",
        "post_url": "https://linkedin.com/feed/update/urn:li:activity:999/",
    }
    tracking.record_seen([post])
    assert tracking.mark_applied(post["post_url"])
    applied = json.loads(tracking.APPLIED_PATH.read_text())
    # The applied key should match the seen urn key
    assert "urn:urn:li:activity:999" in applied


def test_rejected_post_is_filtered_out_next_run():
    post = {"post_urn": "Q", "post_url": "https://linkedin.com/q"}
    tracking.mark_rejected("urn:Q")
    assert tracking.drop_known([post]) == []


def test_applied_post_is_filtered_out_next_run():
    post = {"post_urn": "R", "post_url": "https://linkedin.com/r"}
    tracking.mark_applied("urn:R")
    assert tracking.drop_known([post]) == []


def test_stats_counts():
    tracking.record_seen([{"post_urn": "S1", "post_url": "a"}])
    tracking.mark_applied("urn:X")
    tracking.mark_rejected("urn:Y")
    s = tracking.stats()
    # stats() now exposes `seen_active` (TTL-active keys) alongside total seen.
    # All freshly-recorded entries are active, so counts match.
    assert s["seen"] == 1
    assert s["seen_active"] == 1
    assert s["applied"] == 1
    assert s["rejected"] == 1


def test_corrupted_state_file_is_tolerated():
    tracking.SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    tracking.SEEN_PATH.write_text("{ not valid json")
    # Should not raise
    posts = [{"post_urn": "Z", "post_url": "z"}]
    assert len(tracking.drop_known(posts)) == 1


def test_resolve_key_passes_through_explicit_keys():
    assert tracking._resolve_key("urn:li:activity:123") == "urn:li:activity:123"
    assert tracking._resolve_key("url:abc123") == "url:abc123"


# --------------------- TTL-aware dedup (3-day rolling) ---------------------

from datetime import datetime, timedelta, timezone

from src import config


def _stamp_seen(key: str, delivered_days_ago: float) -> None:
    """Helper: write a synthetic seen entry dated N days in the past."""
    data = tracking._load(tracking.SEEN_PATH)
    ts = (datetime.now(timezone.utc) - timedelta(days=delivered_days_ago)).isoformat()
    data[key] = {"url": f"https://example.com/{key}", "first_seen": ts, "delivered_at": ts}
    tracking._save(tracking.SEEN_PATH, data)


def test_record_seen_writes_delivered_at():
    post = {"post_urn": "T1", "post_url": "https://linkedin.com/t1"}
    tracking.record_seen([post])
    data = tracking._load(tracking.SEEN_PATH)
    entry = data[tracking.post_key(post)]
    assert "delivered_at" in entry
    assert "first_seen" in entry


def test_record_seen_preserves_first_seen_on_re_record():
    """Re-recording a post updates delivered_at but keeps the original first_seen
    (so we can still tell when the user first saw it)."""
    post = {"post_urn": "T2", "post_url": "x"}
    tracking.record_seen([post])
    original = tracking._load(tracking.SEEN_PATH)[tracking.post_key(post)]["first_seen"]
    # Re-record
    tracking.record_seen([post])
    after = tracking._load(tracking.SEEN_PATH)[tracking.post_key(post)]
    assert after["first_seen"] == original
    assert after["delivered_at"] >= original


def test_drop_known_respects_ttl_window(monkeypatch):
    """A post delivered WITHIN the DEDUP_DAYS window is still known;
    a post delivered BEYOND the window is re-eligible."""
    monkeypatch.setattr(config, "DEDUP_DAYS", 3)
    _stamp_seen("urn:fresh", delivered_days_ago=1)   # 1 day ago → still suppressed
    _stamp_seen("urn:stale", delivered_days_ago=5)   # 5 days ago → eligible again

    posts = [
        {"post_urn": "fresh", "post_url": "https://example.com/fresh"},
        {"post_urn": "stale", "post_url": "https://example.com/stale"},
    ]
    out = tracking.drop_known(posts)
    urns = {p["post_urn"] for p in out}
    assert "stale" in urns  # came back into rotation
    assert "fresh" not in urns  # still within 3-day window


def test_drop_known_dedup_disabled_with_zero_days(monkeypatch):
    """DEDUP_DAYS=0 disables TTL; seen_posts are permanently suppressed (legacy)."""
    monkeypatch.setattr(config, "DEDUP_DAYS", 0)
    _stamp_seen("urn:very_old", delivered_days_ago=365)
    posts = [{"post_urn": "very_old", "post_url": "x"}]
    assert tracking.drop_known(posts) == []


def test_applied_and_rejected_ignore_ttl(monkeypatch):
    """User decisions (applied/rejected) are permanent — TTL does NOT expire them."""
    monkeypatch.setattr(config, "DEDUP_DAYS", 3)
    # Use old timestamps by going through the proper API (applied/rejected don't
    # use TTL so even very old entries still dedup).
    tracking.mark_applied("urn:applied_ancient")
    tracking.mark_rejected("urn:rejected_ancient")
    posts = [
        {"post_urn": "applied_ancient", "post_url": "x"},
        {"post_urn": "rejected_ancient", "post_url": "y"},
    ]
    assert tracking.drop_known(posts) == []


def test_prune_seen_removes_only_truly_stale(monkeypatch):
    monkeypatch.setattr(config, "DEDUP_DAYS", 3)  # retain_days = max(12, 30) = 30
    _stamp_seen("urn:recent", delivered_days_ago=5)    # keep
    _stamp_seen("urn:ancient", delivered_days_ago=60)  # prune
    removed = tracking.prune_seen()
    assert removed == 1
    remaining = tracking._load(tracking.SEEN_PATH)
    assert "urn:recent" in remaining
    assert "urn:ancient" not in remaining


def test_active_seen_keys_tolerates_corrupt_entries(monkeypatch):
    """A seen entry missing timestamps or with a non-dict value must not crash
    the active-key scan — it should just be treated as expired."""
    monkeypatch.setattr(config, "DEDUP_DAYS", 3)
    tracking.SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    tracking.SEEN_PATH.write_text(json.dumps({
        "urn:corrupt_string": "legacy_string_value",
        "urn:corrupt_nodate": {"url": "x"},
        "urn:corrupt_baddate": {"url": "x", "delivered_at": "not-an-iso-date"},
        "urn:ok": {"url": "ok", "delivered_at": datetime.now(timezone.utc).isoformat()},
    }))
    active = tracking._active_seen_keys()
    assert active == {"urn:ok"}


def test_legacy_entries_fall_back_to_first_seen(monkeypatch):
    """Pre-TTL records that only have `first_seen` (no `delivered_at`) should
    still participate in the TTL math using first_seen as the anchor."""
    monkeypatch.setattr(config, "DEDUP_DAYS", 3)
    fresh_ts = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    stale_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    tracking.SEEN_PATH.write_text(json.dumps({
        "urn:legacy_fresh": {"url": "x", "first_seen": fresh_ts},
        "urn:legacy_stale": {"url": "y", "first_seen": stale_ts},
    }))
    active = tracking._active_seen_keys()
    assert "urn:legacy_fresh" in active
    assert "urn:legacy_stale" not in active
