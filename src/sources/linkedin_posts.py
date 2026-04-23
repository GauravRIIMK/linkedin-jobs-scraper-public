"""LinkedIn posts source — wraps harvestapi/linkedin-post-search.

This is the original, premium-signal source — #hiring posts with
recruiter context, comments, and apply-method language in body text.
Price: $0.002/post (free-tier). Delegates to `apify_runner.run_query`
so we don't duplicate the actor-call logic.
"""
from __future__ import annotations

from typing import Any

from apify_client import ApifyClient

from .. import apify_runner, config


def scrape(
    client: ApifyClient,
    track_prefs: dict[str, Any],
    queries: list[str],
) -> list[dict[str, Any]]:
    """Run every query for this track; dedupe by URL inside the track."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    cap = config.MAX_POSTS_PER_QUERY
    for q in queries:
        try:
            posts = apify_runner.run_query(client, q, max_posts=cap)
        except Exception as exc:  # noqa: BLE001 — one bad query doesn't kill the rest
            print(f"[linkedin_posts] query {q!r} failed: {exc}")
            continue

        for p in posts:
            url = p.get("post_url")
            if not url or url in seen:
                continue
            seen.add(url)
            # Tag the source so downstream knows how to interpret this record.
            p["source"] = "linkedin_post"
            # linkedin_posts uses free-text apply-method detection, so
            # apply_url is not known up-front.
            p.setdefault("apply_url", None)
            p.setdefault("company", None)
            p.setdefault("location", None)
            out.append(p)

    return out
