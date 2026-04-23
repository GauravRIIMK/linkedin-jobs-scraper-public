"""Apify runner: query input shape, result normalization, dedup, fail-soft."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src import apify_runner


def test_normalize_preserves_core_fields():
    raw = {
        "postUrl": "https://www.linkedin.com/feed/update/urn:li:activity:123/",
        "full_urn": "urn:li:activity:123",
        "postedAt": "2026-04-20T00:00:00Z",
        "text": "Hiring!",
        "author": {"name": "Acme", "title": "Recruiter", "url": "https://linkedin.com/x"},
        "comments": [{"text": "apply here", "author": {"name": "Person"}, "postedAt": "2026-04-20T01:00:00Z"}],
    }
    out = apify_runner._normalize(raw, query="q")
    assert out["post_url"] == raw["postUrl"]
    assert out["post_urn"] == "urn:li:activity:123"
    assert out["author_name"] == "Acme"
    assert out["author_title"] == "Recruiter"
    assert len(out["comments"]) == 1
    assert out["comments"][0]["text"] == "apply here"


def test_normalize_rebuilds_url_from_urn():
    raw = {
        "full_urn": "urn:li:activity:999",
        "text": "hiring",
        "author": {"name": "X"},
    }
    out = apify_runner._normalize(raw, query="q")
    assert out["post_url"].endswith("urn:li:activity:999/")


def test_normalize_handles_missing_author():
    raw = {"postUrl": "https://linkedin.com/x", "text": "hi"}
    out = apify_runner._normalize(raw, query="q")
    assert out["author_name"] is None
    assert out["author_title"] is None


def test_normalize_non_dict_comment_skipped():
    raw = {"postUrl": "https://x", "text": "hi", "comments": ["invalid string comment"]}
    out = apify_runner._normalize(raw, query="q")
    assert out["comments"] == []


def test_run_query_passes_correct_input():
    fake_client = MagicMock()
    fake_run = {"defaultDatasetId": "ds1"}
    fake_client.actor.return_value.call.return_value = fake_run
    fake_client.dataset.return_value.iterate_items.return_value = iter([
        {"postUrl": "https://linkedin.com/a", "text": "hiring", "author": {"name": "x"}}
    ])

    out = apify_runner.run_query(fake_client, query="growth form")

    # Verify the actor ID is correct
    args, kwargs = fake_client.actor.call_args
    assert args[0] == apify_runner.config.APIFY_ACTOR_ID
    # Verify query + date filter
    run_input = fake_client.actor.return_value.call.call_args.kwargs["run_input"]
    assert run_input["searchQueries"] == ["growth form"]
    assert "postedLimitDate" in run_input
    # Must be one of HarvestAPI's allowed enum values
    assert run_input["postedLimit"] in {
        "any", "1h", "24h", "week", "month", "3months", "6months", "year",
    }
    # sortBy must be one of HarvestAPI's enum values
    assert run_input["sortBy"] in {"relevance", "date"}
    assert len(out) == 1


def test_run_query_handles_none_run():
    fake_client = MagicMock()
    fake_client.actor.return_value.call.return_value = None
    assert apify_runner.run_query(fake_client, "q") == []


def test_run_query_handles_empty_dataset():
    fake_client = MagicMock()
    fake_client.actor.return_value.call.return_value = {"defaultDatasetId": "ds"}
    fake_client.dataset.return_value.iterate_items.return_value = iter([])
    assert apify_runner.run_query(fake_client, "q") == []


def test_run_all_dedupes_within_track():
    """Two queries return the same post URL — should only appear once."""
    posts_q1 = [{
        "postUrl": "https://linkedin.com/samepost",
        "text": "hiring",
        "author": {"name": "x"},
    }]
    posts_q2 = [{
        "postUrl": "https://linkedin.com/samepost",
        "text": "hiring",
        "author": {"name": "x"},
    }]

    def fake_call(run_input):
        q = run_input["searchQueries"][0]
        return {"defaultDatasetId": "q1" if "q1" in q else "q2"}

    def fake_iter(ds_id):
        return iter(posts_q1 if ds_id == "q1" else posts_q2)

    with patch("src.apify_runner.ApifyClient") as MockClient:
        client = MockClient.return_value
        client.actor.return_value.call.side_effect = fake_call
        client.dataset.return_value.iterate_items.side_effect = lambda: fake_iter(
            client.dataset.call_args.args[0]
        )
        out = apify_runner.run_all({"track1": ["q1", "q2"]})

    assert len(out["track1"]) == 1


def test_run_all_fail_soft_on_query_exception():
    """One failing query shouldn't kill the whole track."""
    def fake_call(run_input):
        if "broken" in run_input["searchQueries"][0]:
            raise RuntimeError("apify down")
        return {"defaultDatasetId": "ds1"}

    def fake_iter():
        return iter([{"postUrl": "https://linkedin.com/ok", "text": "hiring", "author": {"name": "x"}}])

    with patch("src.apify_runner.ApifyClient") as MockClient:
        client = MockClient.return_value
        client.actor.return_value.call.side_effect = fake_call
        client.dataset.return_value.iterate_items.side_effect = fake_iter
        out = apify_runner.run_all({"t1": ["good_q", "broken_q"]})

    assert len(out["t1"]) == 1


def test_postedLimitDate_is_iso_utc():
    date_str = apify_runner._postedLimitDate()
    assert date_str.endswith("Z")
    # Should be parseable
    from datetime import datetime
    datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")


def test_postedLimitBucket_maps_days_to_enum(monkeypatch):
    """POSTED_LIMIT_DAYS must map to the tightest valid HarvestAPI enum bucket."""
    allowed = {"any", "1h", "24h", "week", "month", "3months", "6months", "year"}
    cases = [
        (1, "24h"),
        (2, "week"),
        (7, "week"),
        (8, "month"),
        (30, "month"),
        (60, "3months"),
        (100, "6months"),
        (200, "year"),
        (500, "any"),
    ]
    for days, expected in cases:
        monkeypatch.setattr(apify_runner.config, "POSTED_LIMIT_DAYS", days)
        got = apify_runner._postedLimitBucket()
        assert got in allowed, f"{got!r} not in allowed set"
        assert got == expected, f"days={days} expected {expected}, got {got}"
