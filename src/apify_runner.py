"""Run the harvestapi/linkedin-post-search actor and collect posts."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from apify_client import ApifyClient

from . import config


def _postedLimitDate() -> str:
    cutoff = datetime.now(timezone.utc) - timedelta(days=config.POSTED_LIMIT_DAYS)
    return cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")


def _postedLimitBucket() -> str:
    """Map POSTED_LIMIT_DAYS to the tightest HarvestAPI enum bucket.

    HarvestAPI accepts only: any | 1h | 24h | week | month | 3months | 6months | year.
    We choose the smallest bucket that still contains our window so we don't
    miss any posts; `postedLimitDate` then post-filters down to the exact cutoff.
    """
    d = int(config.POSTED_LIMIT_DAYS)
    if d <= 1:
        return "24h"
    if d <= 7:
        return "week"
    if d <= 31:
        return "month"
    if d <= 92:
        return "3months"
    if d <= 183:
        return "6months"
    if d <= 366:
        return "year"
    return "any"


def run_query(client: ApifyClient, query: str, max_posts: int | None = None) -> list[dict[str, Any]]:
    """Run one search query against the HarvestAPI actor and return normalized posts."""
    max_posts = max_posts or config.MAX_POSTS_PER_QUERY

    run_input = {
        "searchQueries": [query],
        "postedLimit": _postedLimitBucket(),
        "postedLimitDate": _postedLimitDate(),
        "sortBy": "date",
        "maxPosts": max_posts,
        "startPage": 1,
    }

    run = client.actor(config.APIFY_ACTOR_ID).call(run_input=run_input)
    if run is None:
        return []

    dataset_id = run.get("defaultDatasetId")
    if not dataset_id:
        return []

    items = list(client.dataset(dataset_id).iterate_items())
    return [_normalize(item, query) for item in items]


def _normalize(raw: dict[str, Any], query: str) -> dict[str, Any]:
    """Flatten the raw actor output into a stable shape we control."""
    post_url = (
        raw.get("postUrl")
        or raw.get("url")
        or raw.get("linkedinUrl")
        or _url_from_urn(raw.get("full_urn") or raw.get("urn") or {})
    )
    author = raw.get("author") or {}
    return {
        "source_query": query,
        "post_url": post_url,
        "post_urn": raw.get("full_urn") or (raw.get("urn") or {}).get("activity_urn"),
        "posted_at": raw.get("postedAt") or raw.get("date") or raw.get("postedDate"),
        "text": raw.get("text") or raw.get("postText") or raw.get("content") or "",
        "author_name": author.get("name") if isinstance(author, dict) else str(author),
        "author_title": author.get("title") if isinstance(author, dict) else None,
        "author_url": author.get("url") if isinstance(author, dict) else None,
        "comments": _normalize_comments(raw.get("comments") or []),
        "raw": raw,
    }


def _url_from_urn(urn: Any) -> str | None:
    if isinstance(urn, dict):
        activity = urn.get("activity_urn") or ""
    else:
        activity = str(urn) if urn else ""
    if not activity:
        return None
    if "activity:" in activity:
        activity_id = activity.split("activity:")[-1]
    else:
        activity_id = activity
    return f"https://www.linkedin.com/feed/update/urn:li:activity:{activity_id}/"


def _normalize_comments(items: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in items:
        if not isinstance(c, dict):
            continue
        out.append(
            {
                "text": c.get("text") or c.get("comment") or "",
                "author": (c.get("author") or {}).get("name")
                if isinstance(c.get("author"), dict)
                else c.get("author"),
                "posted_at": c.get("postedAt") or c.get("date"),
            }
        )
    return out


def run_all(queries_by_track: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    """Execute every query, grouped by track. Deduplicate posts within a track by URL."""
    client = ApifyClient(config.APIFY_TOKEN)
    results: dict[str, list[dict[str, Any]]] = {}

    for track, queries in queries_by_track.items():
        seen_urls: set[str] = set()
        track_posts: list[dict[str, Any]] = []
        for q in queries:
            try:
                posts = run_query(client, q)
            except Exception as e:  # noqa: BLE001
                print(f"[apify] query {q!r} failed: {e}")
                continue
            for p in posts:
                url = p.get("post_url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                track_posts.append(p)
        results[track] = track_posts

    return results
