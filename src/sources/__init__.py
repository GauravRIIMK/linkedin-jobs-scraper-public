"""Multi-source scraping orchestrator.

The original pipeline hit a single actor (harvestapi/linkedin-post-search) which
capped out at ~45 unique posts per track per run — the top of the funnel was
starving. This package adds two more data sources that complement it:

  - `linkedin_posts` (current harvestapi, $0.002/post) — the #hiring feed
    with rich comments & recruiter context. Best signal, premium price.

  - `linkedin_jobs` (worldunboxer/rapid-linkedin-scraper, FREE) — LinkedIn's
    structured job board. Apply URLs are known, so these bypass the
    apply-method detector and go straight to matching.

  - `naukri_jobs` (shahidirfan/Naukri-Jobs-Scraper, FREE) — India's #1 job
    board. Huge volume, structured apply URLs.

All three emit the same normalized schema (see `NORMALIZED_SCHEMA_DOC` below)
so every downstream step — filter, match, rerank, deliver — treats them
uniformly. The orchestrator fails soft: one source crashing does not kill
the others.

Budget knobs live in `config.py` — ENABLED_SOURCES, MAX_*_PER_TRACK.

Concurrency: within a single track, the enabled sources run in parallel
(each hits a different Apify actor, so there is no per-actor concurrency
contention). Tracks themselves stay serial because the same actor IS
reused across tracks and paid-plan per-actor concurrency caps are finite.
This alone cuts per-run wall-clock ~3x vs. the original serial fan-out.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from apify_client import ApifyClient

from .. import config
from . import linkedin_jobs, linkedin_posts, naukri_jobs

NORMALIZED_SCHEMA_DOC = """
Every source returns posts in this shape:

    {
        "source": "linkedin_post" | "linkedin_job" | "naukri",
        "post_url": str,           # canonical URL to the listing/post
        "post_urn": str | None,    # LinkedIn post URN when applicable
        "text": str,               # post body or JD text for matching
        "author_name": str,        # recruiter or company
        "author_title": str | None,# recruiter title or role tagline
        "posted_at": str | None,   # ISO date or source-specific string
        "comments": list[dict],    # only populated for linkedin_post
        # job-board extras (optional, present for job sources):
        "apply_url": str | None,   # direct apply URL when known up-front
        "company": str | None,
        "location": str | None,
        "source_query": str,       # the query that yielded this record
        "raw": dict,               # original actor output for debugging
    }
"""


def _source_modules() -> dict[str, Any]:
    return {
        "linkedin_posts": linkedin_posts,
        "linkedin_jobs": linkedin_jobs,
        "naukri_jobs": naukri_jobs,
    }


def _run_source_safe(
    source_name: str,
    module: Any,
    client: ApifyClient,
    track_prefs: dict[str, Any],
    queries: list[str],
) -> list[dict[str, Any]]:
    """Call one source's scrape() and swallow any exception into an empty list.

    Isolated helper so ThreadPoolExecutor.submit has a plain function to call
    and so every future completes with a list (never an exception) — the
    orchestrator then merges without further try/except.
    """
    try:
        return module.scrape(client=client, track_prefs=track_prefs, queries=queries)
    except Exception as exc:  # noqa: BLE001 — per-source failure is non-fatal
        track = track_prefs.get("track", "?")
        print(f"[sources] {source_name} failed for track {track}: {exc}")
        return []


def scrape_all(
    prefs: dict[str, Any],
    queries_by_track: dict[str, list[str]],
) -> dict[str, list[dict[str, Any]]]:
    """Fan out to every enabled source; merge per-track results with URL dedup.

    - `prefs` is the full role_preferences dict (so naukri/linkedin_jobs can
       read role.specific from track_prefs, not just the query strings).
    - `queries_by_track` is the LinkedIn-post query ladder from query_builder;
       only linkedin_posts consumes it, because job boards don't understand
       hashtag search.

    Within each track the enabled sources run in parallel (different Apify
    actors, safe to overlap). Tracks themselves stay serial to avoid
    stacking multiple concurrent runs of the SAME actor — per-actor
    concurrency limits can cap us at 3-5 simultaneous runs.

    Graceful partial-drain: if `config.SCRAPE_SOFT_DEADLINE_SECONDS`
    elapses before all tracks have been visited, we stop starting new
    tracks and return whatever was collected so far. The rest of the
    pipeline (filter → match → rerank → deliver) then runs normally on
    the partial dict — tracks that completed still ship matches to the
    email; tracks that didn't start appear as empty. Much better than
    having the GitHub Actions hard-timeout kill the whole process and
    throw away every scraped post.

    Returns {track: [normalized_post, ...]} across all sources. Within a
    track, URL dedup tie-breaks in favor of the source that appears first in
    `config.ENABLED_SOURCES` — so swap `linkedin_posts` ahead of the job
    boards if you want post-text wins over job-board text for shared URLs.
    """
    client = ApifyClient(config.APIFY_TOKEN)
    enabled = [s.strip() for s in config.ENABLED_SOURCES if s.strip()]
    modules = _source_modules()
    # Preserve the configured priority order for dedup tie-break.
    enabled_modules: list[tuple[str, Any]] = [
        (name, modules[name]) for name in enabled if name in modules
    ]

    merged: dict[str, list[dict[str, Any]]] = {}
    track_prefs_by_name = {t["track"]: t for t in prefs.get("tracks", [])}

    start = time.monotonic()
    deadline = config.SCRAPE_SOFT_DEADLINE_SECONDS
    track_items = list(queries_by_track.items())

    for idx, (track, queries) in enumerate(track_items):
        # Soft-deadline check: if we've run past the budget, give up on
        # the remaining tracks but let everything collected so far flow
        # downstream. `idx > 0` ensures at least one track always gets a
        # chance even on pathologically slow starts.
        elapsed = time.monotonic() - start
        if idx > 0 and elapsed >= deadline:
            remaining = [t for t, _ in track_items[idx:]]
            print(
                f"[sources] Soft deadline {deadline}s reached after {elapsed:.0f}s; "
                f"skipping remaining tracks {remaining} and returning partial data."
            )
            for t in remaining:
                merged.setdefault(t, [])
            break

        tp = track_prefs_by_name.get(track, {"track": track, "roles": []})
        source_results: dict[str, list[dict[str, Any]]] = {}

        if not enabled_modules:
            merged[track] = []
            continue

        with ThreadPoolExecutor(max_workers=len(enabled_modules)) as ex:
            futures = {
                ex.submit(
                    _run_source_safe, name, module, client, tp, queries
                ): name
                for name, module in enabled_modules
            }
            for fut in as_completed(futures):
                source_results[futures[fut]] = fut.result()

        # Merge in the configured source order so the dedup tie-break is
        # deterministic and independent of which future completed first.
        seen_urls: set[str] = set()
        merged[track] = []
        for name, _ in enabled_modules:
            for p in source_results.get(name, []):
                url = p.get("post_url")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                merged[track].append(p)

    return merged


def source_distribution(
    posts_by_track: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    """Per-track breakdown of which source each post came from. For logging."""
    out: dict[str, dict[str, int]] = {}
    for track, posts in posts_by_track.items():
        bucket: dict[str, int] = {}
        for p in posts:
            s = p.get("source", "unknown")
            bucket[s] = bucket.get(s, 0) + 1
        out[track] = bucket
    return out
