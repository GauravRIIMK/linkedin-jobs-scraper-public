"""LinkedIn Jobs source — worldunboxer/rapid-linkedin-scraper (FREE).

LinkedIn's structured job-board is a fundamentally different corpus from
the #hiring feed we already scrape: it has up-to-date apply URLs, company
metadata, and no ambiguity about whether the post is hiring. It's also
FREE on Apify, which matters because the `$5/month Apify budget` doesn't
leave much room to expand the paid harvestapi source.

Input contract (from actor schema — confirmed via fetch-actor-details):
    job_title: str
    location: str
    jobs_entries: int           # cap per call
    job_post_time: "r86400" | "r604800" | "r2592000"   (24h | 7d | 30d)
    experience_level, job_type, work_schedule, company_names  (optional)

Strategy: one call per (role × location) tuple per track. Dedup by
jobUrl/jobId is handled in the orchestrator.

If the actor changes its output schema, the normalizer below degrades
gracefully: anything missing becomes None/"" and the post still flows
into the pipeline (where similarity scoring and Haiku will filter it).
"""
from __future__ import annotations

from typing import Any

from apify_client import ApifyClient

from .. import config

ACTOR_ID = "worldunboxer/rapid-linkedin-scraper"

# Map POSTED_LIMIT_DAYS → actor's bucketed enum. Only 3 buckets supported,
# so we round UP to the next bucket that still contains our window.
_TIME_BUCKETS = [
    (1, "r86400"),     # past 24h
    (7, "r604800"),    # past week
    (30, "r2592000"),  # past month
]


def _time_bucket() -> str:
    d = int(config.POSTED_LIMIT_DAYS)
    for threshold, code in _TIME_BUCKETS:
        if d <= threshold:
            return code
    return "r2592000"


def _role_candidates(track_prefs: dict[str, Any]) -> list[str]:
    """Pick the top K roles for this track. Job boards respect full titles,
    so we use each role verbatim (not the narrowed-down keyword used for
    LinkedIn post hashtag search)."""
    roles = [r for r in (track_prefs.get("roles") or []) if r.strip()]
    return roles[: config.ROLES_PER_JOB_BOARD_TRACK]


def _location_candidates(track_prefs: dict[str, Any]) -> list[str]:
    """Which locations to query. By default we blanket "India" so we don't
    over-narrow. Remote gets its own call when the track opts in, so
    remote-heavy hits aren't drowned out by India-only searches."""
    locs = ["India"]
    if track_prefs.get("include_remote", True):
        locs.append("Remote")
    return locs


def _call_actor(
    client: ApifyClient,
    job_title: str,
    location: str,
    jobs_entries: int,
) -> list[dict[str, Any]]:
    run_input = {
        "job_title": job_title,
        "location": location,
        "jobs_entries": jobs_entries,
        "job_post_time": _time_bucket(),
        "start_jobs": 0,
    }
    run = client.actor(ACTOR_ID).call(run_input=run_input)
    if run is None:
        return []
    dataset_id = run.get("defaultDatasetId")
    if not dataset_id:
        return []
    return list(client.dataset(dataset_id).iterate_items())


def _normalize(raw: dict[str, Any], query: str) -> dict[str, Any]:
    """Actor output schema isn't rigid; probe common field names so a rename
    upstream doesn't silently zero out our text/url. Matching only needs
    `text` and `post_url` to be non-empty; everything else is icing."""
    title = raw.get("job_title") or raw.get("title") or raw.get("jobTitle") or ""
    company = (
        raw.get("company_name")
        or raw.get("company")
        or (raw.get("company_info") or {}).get("name")
        or ""
    )
    description = (
        raw.get("job_description")
        or raw.get("description")
        or raw.get("jobDescription")
        or ""
    )
    job_url = (
        raw.get("job_url")
        or raw.get("jobUrl")
        or raw.get("url")
        or raw.get("linkedinUrl")
        or ""
    )
    apply_url = (
        raw.get("apply_url")
        or raw.get("applyUrl")
        or raw.get("application_url")
        or job_url  # LinkedIn job URL is an acceptable apply target
    )
    location = raw.get("location") or raw.get("job_location") or ""
    posted_at = (
        raw.get("posted_at")
        or raw.get("postedAt")
        or raw.get("listed_at")
        or raw.get("date")
        or ""
    )
    # Fuse title+company into a synthetic author_name so CSV/HTML rows
    # still look populated (recruiter column on LinkedIn posts has a
    # human name; for jobs we substitute "<Company> (Jobs)").
    author_name = company or "Unknown"
    author_title = title or ""
    # Matcher needs meaningful text; concatenate the strongest fields.
    text = "\n".join(x for x in (title, company, location, description) if x)

    return {
        "source": "linkedin_job",
        "post_url": job_url,
        "post_urn": raw.get("job_id") or raw.get("id"),
        "text": text,
        "author_name": author_name,
        "author_title": author_title,
        "posted_at": posted_at,
        "comments": [],
        "apply_url": apply_url,
        "company": company or None,
        "location": location or None,
        "source_query": query,
        "raw": raw,
    }


def scrape(
    client: ApifyClient,
    track_prefs: dict[str, Any],
    queries: list[str],  # unused for job boards, kept for signature parity
) -> list[dict[str, Any]]:
    if not config.MAX_LINKEDIN_JOBS_PER_TRACK:
        return []

    roles = _role_candidates(track_prefs)
    locs = _location_candidates(track_prefs)
    if not roles:
        return []

    # Split the per-track budget evenly across (roles × locations) so no
    # single role monopolizes the call quota.
    n_calls = max(1, len(roles) * len(locs))
    per_call = max(25, config.MAX_LINKEDIN_JOBS_PER_TRACK // n_calls)

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for role in roles:
        for loc in locs:
            try:
                raws = _call_actor(client, role, loc, per_call)
            except Exception as exc:  # noqa: BLE001
                print(f"[linkedin_jobs] {role!r}@{loc!r} failed: {exc}")
                continue
            q = f"{role} ({loc})"
            for r in raws:
                p = _normalize(r, q)
                url = p["post_url"]
                if not url or url in seen:
                    continue
                seen.add(url)
                out.append(p)

    return out
