"""Multi-source scraping: normalizers, dedup, fail-soft, orchestrator."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src import sources
from src.sources import linkedin_jobs, linkedin_posts, naukri_jobs


# ---------------------------------------------------------------------------
# linkedin_jobs
# ---------------------------------------------------------------------------

def test_linkedin_jobs_time_bucket_respects_days(monkeypatch):
    allowed = {"r86400", "r604800", "r2592000"}
    cases = [
        (1, "r86400"),
        (2, "r604800"),
        (7, "r604800"),
        (8, "r2592000"),
        (30, "r2592000"),
        (365, "r2592000"),
    ]
    for days, expected in cases:
        monkeypatch.setattr(linkedin_jobs.config, "POSTED_LIMIT_DAYS", days)
        got = linkedin_jobs._time_bucket()
        assert got in allowed
        assert got == expected, f"days={days} expected {expected} got {got}"


def test_linkedin_jobs_normalize_probes_multiple_field_names():
    """Actor output schema isn't rigid; we should still pull the core fields."""
    raw = {
        "jobTitle": "Senior PM",
        "company": "Acme",
        "jobDescription": "Build stuff",
        "linkedinUrl": "https://linkedin.com/jobs/view/42",
        "applyUrl": "https://acme.com/apply/42",
        "location": "Bangalore",
        "postedAt": "2026-04-20",
        "id": "42",
    }
    out = linkedin_jobs._normalize(raw, query="PM (India)")
    assert out["source"] == "linkedin_job"
    assert out["post_url"] == "https://linkedin.com/jobs/view/42"
    assert out["apply_url"] == "https://acme.com/apply/42"
    assert out["company"] == "Acme"
    assert out["author_title"] == "Senior PM"
    # Text blob feeds the matcher — must concatenate the useful fields.
    assert "Senior PM" in out["text"]
    assert "Acme" in out["text"]
    assert "Build stuff" in out["text"]
    assert out["source_query"] == "PM (India)"


def test_linkedin_jobs_normalize_falls_back_to_job_url_as_apply():
    """When no explicit apply_url is present, the job URL is acceptable."""
    raw = {"job_title": "Eng", "job_url": "https://linkedin.com/jobs/view/7"}
    out = linkedin_jobs._normalize(raw, query="Eng")
    assert out["apply_url"] == "https://linkedin.com/jobs/view/7"


def test_linkedin_jobs_scrape_zero_budget_returns_empty(monkeypatch):
    monkeypatch.setattr(linkedin_jobs.config, "MAX_LINKEDIN_JOBS_PER_TRACK", 0)
    assert linkedin_jobs.scrape(client=MagicMock(), track_prefs={"roles": ["PM"]}, queries=[]) == []


def test_linkedin_jobs_scrape_no_roles_returns_empty():
    assert linkedin_jobs.scrape(client=MagicMock(), track_prefs={"roles": []}, queries=[]) == []


def test_linkedin_jobs_scrape_fail_soft_on_role_exception(monkeypatch):
    """One role's actor call dies — the others still flow through."""
    monkeypatch.setattr(linkedin_jobs.config, "ROLES_PER_JOB_BOARD_TRACK", 2)
    monkeypatch.setattr(linkedin_jobs.config, "MAX_LINKEDIN_JOBS_PER_TRACK", 100)

    def fake_call(client, job_title, location, jobs_entries):
        if job_title == "Broken":
            raise RuntimeError("actor down")
        return [{"job_title": job_title, "job_url": f"https://x/{job_title}"}]

    with patch.object(linkedin_jobs, "_call_actor", side_effect=fake_call):
        out = linkedin_jobs.scrape(
            client=MagicMock(),
            track_prefs={"roles": ["Broken", "OK"], "include_remote": False},
            queries=[],
        )

    urls = [p["post_url"] for p in out]
    assert any("OK" in u for u in urls)
    assert not any("Broken" in u for u in urls)


# ---------------------------------------------------------------------------
# naukri_jobs
# ---------------------------------------------------------------------------

def test_naukri_normalize_core_fields():
    raw = {
        "jobTitle": "Growth Manager",
        "companyName": "Acme India",
        "description": "Growth role at Acme",
        "jobUrl": "https://naukri.com/job/123",
        "jobLocation": "Bengaluru",
        "experienceRange": "3-5 years",
        "salaryLabel": "15-25 LPA",
        "skills": ["Growth", "SQL", "Python"],
        "postedAt": "2026-04-19",
        "jobId": "123",
    }
    out = naukri_jobs._normalize(raw, query="Growth Manager")
    assert out["source"] == "naukri"
    assert out["post_url"] == "https://naukri.com/job/123"
    assert out["apply_url"] == "https://naukri.com/job/123"  # listing IS apply
    assert out["company"] == "Acme India"
    assert "Growth" in out["text"]
    assert "SQL" in out["text"]  # skill tokens inline


def test_naukri_normalize_handles_missing_fields():
    raw = {"jobUrl": "https://naukri.com/x"}
    out = naukri_jobs._normalize(raw, query="q")
    assert out["author_name"] == "Unknown"
    assert out["text"] == ""  # nothing to concatenate


def test_naukri_scrape_deduplicates_by_url(monkeypatch):
    monkeypatch.setattr(naukri_jobs.config, "ROLES_PER_JOB_BOARD_TRACK", 2)
    monkeypatch.setattr(naukri_jobs.config, "MAX_NAUKRI_JOBS_PER_TRACK", 100)
    dup = {"jobTitle": "PM", "jobUrl": "https://naukri.com/same"}

    with patch.object(naukri_jobs, "_call_actor", return_value=[dup, dup]):
        out = naukri_jobs.scrape(
            client=MagicMock(),
            track_prefs={"roles": ["PM", "PM2"]},
            queries=[],
        )

    assert len(out) == 1


# ---------------------------------------------------------------------------
# linkedin_posts wrapper
# ---------------------------------------------------------------------------

def test_linkedin_posts_wraps_runner_and_tags_source():
    fake_posts = [
        {"post_url": "https://linkedin.com/a", "text": "hi"},
        {"post_url": "https://linkedin.com/b", "text": "hi"},
    ]
    with patch.object(linkedin_posts.apify_runner, "run_query", return_value=fake_posts):
        out = linkedin_posts.scrape(
            client=MagicMock(),
            track_prefs={"roles": ["x"]},
            queries=["q1", "q2"],
        )

    assert len(out) == 2  # dedup across queries
    assert all(p["source"] == "linkedin_post" for p in out)
    assert all("apply_url" in p for p in out)


def test_linkedin_posts_fail_soft_on_query_error():
    def fake(client, q, max_posts=None):
        if q == "boom":
            raise RuntimeError("actor down")
        return [{"post_url": f"https://x/{q}", "text": "hi"}]

    with patch.object(linkedin_posts.apify_runner, "run_query", side_effect=fake):
        out = linkedin_posts.scrape(
            client=MagicMock(),
            track_prefs={"roles": ["x"]},
            queries=["ok", "boom", "also_ok"],
        )

    urls = {p["post_url"] for p in out}
    assert "https://x/ok" in urls
    assert "https://x/also_ok" in urls


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def test_scrape_all_merges_and_dedupes_across_sources(monkeypatch):
    """Same URL from two different sources should appear once in the merged list."""
    monkeypatch.setattr(
        sources.config, "ENABLED_SOURCES",
        ["linkedin_posts", "linkedin_jobs", "naukri_jobs"],
    )

    def fake_linkedin_posts(**kw):
        return [{"source": "linkedin_post", "post_url": "https://shared/1", "text": "a"}]

    def fake_linkedin_jobs(**kw):
        return [
            {"source": "linkedin_job", "post_url": "https://shared/1", "text": "b"},
            {"source": "linkedin_job", "post_url": "https://only-lj/2", "text": "c"},
        ]

    def fake_naukri(**kw):
        return [{"source": "naukri", "post_url": "https://only-n/3", "text": "d"}]

    with patch.object(sources.linkedin_posts, "scrape", side_effect=fake_linkedin_posts), \
         patch.object(sources.linkedin_jobs, "scrape", side_effect=fake_linkedin_jobs), \
         patch.object(sources.naukri_jobs, "scrape", side_effect=fake_naukri):
        prefs = {"tracks": [{"track": "pm", "roles": ["PM"]}]}
        out = sources.scrape_all(prefs, {"pm": ["#hiring PM"]})

    urls = {p["post_url"] for p in out["pm"]}
    assert urls == {"https://shared/1", "https://only-lj/2", "https://only-n/3"}


def test_scrape_all_respects_enabled_sources(monkeypatch):
    """Dropping a source from ENABLED_SOURCES must skip its scrape call."""
    monkeypatch.setattr(sources.config, "ENABLED_SOURCES", ["linkedin_posts"])

    with patch.object(sources.linkedin_posts, "scrape", return_value=[
        {"source": "linkedin_post", "post_url": "https://a/1", "text": "x"}
    ]) as lp, \
         patch.object(sources.linkedin_jobs, "scrape") as lj, \
         patch.object(sources.naukri_jobs, "scrape") as nj:
        prefs = {"tracks": [{"track": "pm", "roles": ["PM"]}]}
        sources.scrape_all(prefs, {"pm": ["q"]})

    lp.assert_called_once()
    lj.assert_not_called()
    nj.assert_not_called()


def test_scrape_all_fail_soft_on_source_crash(monkeypatch):
    """One source blowing up should not starve the other tracks/sources."""
    monkeypatch.setattr(
        sources.config, "ENABLED_SOURCES",
        ["linkedin_posts", "linkedin_jobs", "naukri_jobs"],
    )

    with patch.object(sources.linkedin_posts, "scrape",
                      side_effect=RuntimeError("apify 500")), \
         patch.object(sources.linkedin_jobs, "scrape", return_value=[
             {"source": "linkedin_job", "post_url": "https://x/1", "text": "a"}
         ]), \
         patch.object(sources.naukri_jobs, "scrape", return_value=[
             {"source": "naukri", "post_url": "https://x/2", "text": "b"}
         ]):
        prefs = {"tracks": [{"track": "pm", "roles": ["PM"]}]}
        out = sources.scrape_all(prefs, {"pm": ["q"]})

    urls = {p["post_url"] for p in out["pm"]}
    assert urls == {"https://x/1", "https://x/2"}


def test_source_distribution_counts_by_source():
    out = sources.source_distribution({
        "pm": [
            {"source": "linkedin_post", "post_url": "a"},
            {"source": "linkedin_post", "post_url": "b"},
            {"source": "naukri", "post_url": "c"},
        ],
        "growth": [
            {"source": "linkedin_job", "post_url": "d"},
        ],
    })
    assert out["pm"] == {"linkedin_post": 2, "naukri": 1}
    assert out["growth"] == {"linkedin_job": 1}


# ---------------------------------------------------------------------------
# Orchestrator: parallelism contracts
# ---------------------------------------------------------------------------


def test_scrape_all_runs_sources_in_parallel(monkeypatch):
    """The three sources should run overlapping, not strictly back-to-back.

    We prove this by making each source block on a barrier sized to all
    three. If scrape_all were calling them serially, the first source would
    wait forever — no second arrival, barrier never releases. If it's
    parallel, all three arrive and release together.
    """
    import threading
    monkeypatch.setattr(
        sources.config, "ENABLED_SOURCES",
        ["linkedin_posts", "linkedin_jobs", "naukri_jobs"],
    )

    barrier = threading.Barrier(3, timeout=5)

    def blocking(source_tag):
        def _inner(**kw):
            barrier.wait()  # blocks until all 3 sources hit this line
            return [{"source": source_tag, "post_url": f"https://x/{source_tag}", "text": "."}]
        return _inner

    with patch.object(sources.linkedin_posts, "scrape", side_effect=blocking("linkedin_post")), \
         patch.object(sources.linkedin_jobs, "scrape", side_effect=blocking("linkedin_job")), \
         patch.object(sources.naukri_jobs, "scrape", side_effect=blocking("naukri")):
        prefs = {"tracks": [{"track": "pm", "roles": ["PM"]}]}
        out = sources.scrape_all(prefs, {"pm": ["q"]})

    # All three arrived at the barrier and each produced exactly one post.
    assert len(out["pm"]) == 3


def test_scrape_all_dedup_order_follows_enabled_sources(monkeypatch):
    """When the same URL comes from multiple sources, the FIRST source
    listed in ENABLED_SOURCES must win, regardless of future completion
    order. We simulate out-of-order completion by making linkedin_posts
    slow and naukri fast — dedup should still pick linkedin_post as the
    survivor for the shared URL because it's listed first."""
    import time
    monkeypatch.setattr(
        sources.config, "ENABLED_SOURCES",
        ["linkedin_posts", "linkedin_jobs", "naukri_jobs"],
    )

    def slow_linkedin(**kw):
        time.sleep(0.05)
        return [{"source": "linkedin_post", "post_url": "https://shared", "text": "LP"}]

    def fast_naukri(**kw):
        return [{"source": "naukri", "post_url": "https://shared", "text": "NK"}]

    with patch.object(sources.linkedin_posts, "scrape", side_effect=slow_linkedin), \
         patch.object(sources.linkedin_jobs, "scrape", return_value=[]), \
         patch.object(sources.naukri_jobs, "scrape", side_effect=fast_naukri):
        prefs = {"tracks": [{"track": "pm", "roles": ["PM"]}]}
        out = sources.scrape_all(prefs, {"pm": ["q"]})

    assert len(out["pm"]) == 1
    # linkedin_posts wins the dedup tie-break even though naukri finished first
    assert out["pm"][0]["source"] == "linkedin_post"
    assert out["pm"][0]["text"] == "LP"


def test_scrape_all_empty_enabled_sources_returns_empty_per_track(monkeypatch):
    """Edge: if someone sets ENABLED_SOURCES="" nothing should crash."""
    monkeypatch.setattr(sources.config, "ENABLED_SOURCES", [])
    prefs = {"tracks": [{"track": "pm", "roles": ["PM"]}]}
    out = sources.scrape_all(prefs, {"pm": ["q"]})
    assert out == {"pm": []}


# ---------------------------------------------------------------------------
# Orchestrator: soft deadline / graceful partial drain
# ---------------------------------------------------------------------------


def test_scrape_all_stops_after_soft_deadline(monkeypatch):
    """When the scrape soft-deadline is exceeded, remaining tracks are
    skipped but whatever was collected still flows through.

    We fake a deadline of 0 seconds — the FIRST track always runs (the
    `idx > 0` guard guarantees forward progress even on pathologically
    tight deadlines), but every subsequent track is dropped with an
    empty list so the pipeline sees the shape it expects."""
    monkeypatch.setattr(
        sources.config, "ENABLED_SOURCES",
        ["linkedin_posts", "linkedin_jobs", "naukri_jobs"],
    )
    monkeypatch.setattr(sources.config, "SCRAPE_SOFT_DEADLINE_SECONDS", 0)

    def fake_lp(**kw):
        track = kw["track_prefs"]["track"]
        return [{"source": "linkedin_post", "post_url": f"https://x/{track}", "text": "."}]

    with patch.object(sources.linkedin_posts, "scrape", side_effect=fake_lp), \
         patch.object(sources.linkedin_jobs, "scrape", return_value=[]), \
         patch.object(sources.naukri_jobs, "scrape", return_value=[]):
        prefs = {"tracks": [
            {"track": "a", "roles": ["A"]},
            {"track": "b", "roles": ["B"]},
            {"track": "c", "roles": ["C"]},
        ]}
        out = sources.scrape_all(prefs, {"a": ["q"], "b": ["q"], "c": ["q"]})

    # First track completes, b and c are listed but empty — the pipeline
    # downstream can still iterate the dict without KeyErrors.
    assert len(out["a"]) == 1
    assert out["b"] == []
    assert out["c"] == []


def test_scrape_all_does_not_skip_first_track_even_with_tight_deadline(monkeypatch):
    """Even with SCRAPE_SOFT_DEADLINE_SECONDS=0, the first track runs —
    otherwise a tight deadline would starve the funnel completely."""
    monkeypatch.setattr(sources.config, "ENABLED_SOURCES", ["linkedin_posts"])
    monkeypatch.setattr(sources.config, "SCRAPE_SOFT_DEADLINE_SECONDS", 0)

    with patch.object(sources.linkedin_posts, "scrape", return_value=[
        {"source": "linkedin_post", "post_url": "https://x/1", "text": "."}
    ]):
        prefs = {"tracks": [{"track": "only", "roles": ["X"]}]}
        out = sources.scrape_all(prefs, {"only": ["q"]})

    assert len(out["only"]) == 1


def test_scrape_all_generous_deadline_does_not_short_circuit(monkeypatch):
    """Sanity: with a huge deadline, no track is skipped — regression
    guard in case the deadline logic ever fires spuriously."""
    monkeypatch.setattr(sources.config, "ENABLED_SOURCES", ["linkedin_posts"])
    monkeypatch.setattr(sources.config, "SCRAPE_SOFT_DEADLINE_SECONDS", 3600)

    def fake_lp(**kw):
        track = kw["track_prefs"]["track"]
        return [{"source": "linkedin_post", "post_url": f"https://x/{track}", "text": "."}]

    with patch.object(sources.linkedin_posts, "scrape", side_effect=fake_lp):
        prefs = {"tracks": [
            {"track": "a", "roles": ["A"]},
            {"track": "b", "roles": ["B"]},
        ]}
        out = sources.scrape_all(prefs, {"a": ["q"], "b": ["q"]})

    assert len(out["a"]) == 1
    assert len(out["b"]) == 1
