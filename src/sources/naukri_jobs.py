"""Naukri Jobs source — shahidirfan/Naukri-Jobs-Scraper (FREE).

Naukri is India's largest job board (75M+ users). Most of the scrapes
we want on a $5/month budget will come from here because it's FREE on
Apify, high-volume, and indexes the long tail that LinkedIn #hiring
posts miss — especially mid-tier Indian employers and agency postings.

Input (confirmed via fetch-actor-details):
    searchQuery: str (role keyword)
    location:    str (single city OR empty → pan-India)
    maxJobs:     int (0 = unlimited)
    experience:  "all" | "0-1" | "1-3" | "3-5" | "5-10" | "10+"
    jobType:     "all" | "full-time" | "part-time" | "contract" | "internship"
    proxyConfiguration: {"useApifyProxy": true}  (strongly recommended)

Strategy: one call per role with pan-India location. Naukri's own search
fans across Indian cities without us having to loop them.
"""
from __future__ import annotations

from typing import Any

from apify_client import ApifyClient

from .. import config

ACTOR_ID = "shahidirfan/Naukri-Jobs-Scraper"


def _roles(track_prefs: dict[str, Any]) -> list[str]:
    roles = [r for r in (track_prefs.get("roles") or []) if r.strip()]
    return roles[: config.ROLES_PER_JOB_BOARD_TRACK]


def _call_actor(
    client: ApifyClient,
    search_query: str,
    max_jobs: int,
) -> list[dict[str, Any]]:
    run_input = {
        "searchQuery": search_query,
        "location": "",  # empty = pan-India
        "maxJobs": max_jobs,
        "experience": "all",
        "jobType": "all",
        "proxyConfiguration": {"useApifyProxy": True},
    }
    run = client.actor(ACTOR_ID).call(run_input=run_input)
    if run is None:
        return []
    dataset_id = run.get("defaultDatasetId")
    if not dataset_id:
        return []
    return list(client.dataset(dataset_id).iterate_items())


def _normalize(raw: dict[str, Any], query: str) -> dict[str, Any]:
    """Naukri field names are snake_case or camelCase depending on actor
    version — probe both. We only NEED text + post_url for the matcher
    to work; everything else is for display."""
    title = raw.get("title") or raw.get("jobTitle") or raw.get("job_title") or ""
    company = (
        raw.get("companyName")
        or raw.get("company_name")
        or raw.get("company")
        or ""
    )
    description = (
        raw.get("description")
        or raw.get("job_description")
        or raw.get("plainDescription")
        or raw.get("descriptionText")
        or ""
    )
    job_url = (
        raw.get("jobUrl")
        or raw.get("job_url")
        or raw.get("url")
        or raw.get("absolute_url")
        or ""
    )
    location = raw.get("location") or raw.get("jobLocation") or ""
    experience = raw.get("experience") or raw.get("experienceRange") or ""
    salary = raw.get("salary") or raw.get("salaryLabel") or ""
    skills = raw.get("skills") or raw.get("tags") or []
    posted_at = (
        raw.get("postedAt")
        or raw.get("posted_at")
        or raw.get("createdAt")
        or raw.get("listed_at")
        or ""
    )

    # Concatenate the richest textual signal we can for embedding.
    skill_text = ", ".join(skills) if isinstance(skills, list) else str(skills or "")
    text = "\n".join(
        x for x in (title, company, location, experience, salary, skill_text, description) if x
    )

    return {
        "source": "naukri",
        "post_url": job_url,
        "post_urn": raw.get("jobId") or raw.get("job_id") or raw.get("id"),
        "text": text,
        "author_name": company or "Unknown",
        "author_title": title or "",
        "posted_at": posted_at,
        "comments": [],
        "apply_url": job_url,  # Naukri's listing URL IS the apply target
        "company": company or None,
        "location": location or None,
        "source_query": query,
        "raw": raw,
    }


def scrape(
    client: ApifyClient,
    track_prefs: dict[str, Any],
    queries: list[str],  # unused for job boards
) -> list[dict[str, Any]]:
    if not config.MAX_NAUKRI_JOBS_PER_TRACK:
        return []

    roles = _roles(track_prefs)
    if not roles:
        return []

    per_call = max(25, config.MAX_NAUKRI_JOBS_PER_TRACK // max(1, len(roles)))

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for role in roles:
        try:
            raws = _call_actor(client, role, per_call)
        except Exception as exc:  # noqa: BLE001
            print(f"[naukri_jobs] {role!r} failed: {exc}")
            continue
        for r in raws:
            p = _normalize(r, role)
            url = p["post_url"]
            if not url or url in seen:
                continue
            seen.add(url)
            out.append(p)

    return out
