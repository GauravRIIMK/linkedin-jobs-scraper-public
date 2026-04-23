"""End-to-end pipeline with all external dependencies mocked:
Apify → gform filter → JD extractor (no network fetches) → matcher → reranker → dedup → delivery.
Covers: dry-run mode, fail-soft, cross-track dedup, empty-state runs.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np

from src import config, pipeline, role_selector


def _write_prefs(tmp_path):
    prefs = {
        "tracks": [
            {
                "track": "growth_marketing",
                "active": True,
                "roles": ["Growth Marketer"],
                "cities": ["Bangalore"],
                "include_remote": True,
                "resume_summary": "Growth marketer with 6 yrs SaaS / Fintech experience; PLG, SEO, paid, lifecycle.",
            },
            {
                "track": "product_ai",
                "active": True,
                "roles": ["AI Product Manager"],
                "cities": ["Bangalore"],
                "include_remote": True,
                "resume_summary": "Product manager with 6 yrs; ML / AI strategy, LLM apps, experimentation.",
            },
        ]
    }
    config.PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.PREFS_PATH.write_text(json.dumps(prefs))


def _apify_posts(track: str) -> list[dict]:
    if track == "growth_marketing":
        return [
            {
                "postUrl": "https://linkedin.com/growth/1",
                "full_urn": "urn:li:activity:G1",
                "text": "Hiring Growth Marketer! Apply: https://forms.gle/GROWTH1 — SaaS, Bangalore, 4+ yrs.",
                "author": {"name": "Acme", "title": "Recruiter"},
                "comments": [],
            },
            {
                "postUrl": "https://linkedin.com/growth/2-nogform",
                "full_urn": "urn:li:activity:G2",
                "text": "Hiring Growth Marketer — DM to apply.",
                "author": {"name": "Beta Corp", "title": "Founder"},
                "comments": [],
            },
        ]
    if track == "product_ai":
        return [
            {
                "postUrl": "https://linkedin.com/pm/1",
                "full_urn": "urn:li:activity:P1",
                "text": "AI PM wanted. Apply via https://forms.gle/PM1. LLM apps experience preferred.",
                "author": {"name": "Gamma", "title": "CPO"},
                "comments": [],
            },
        ]
    return []


def test_pipeline_end_to_end_dry_run(tmp_path):
    _write_prefs(tmp_path)

    def fake_scrape(prefs, queries_by_track):
        return {t: [_normalize_raw(r) for r in _apify_posts(t)] for t in queries_by_track}

    # Mock Voyage: every text gets a unit vector
    def fake_embed(texts, input_type):
        return np.ones((len(texts), 4), dtype=np.float32)

    # Mock Haiku score: always 8.0
    def fake_score_one(track, summary, post, client=None):
        return {"score": 8.0, "reason": "good fit"}

    with patch("src.pipeline.sources.scrape_all", side_effect=fake_scrape), \
         patch("src.matcher.embed_texts", side_effect=fake_embed), \
         patch("src.reranker.score_one", side_effect=fake_score_one), \
         patch("src.pipeline.Anthropic"), \
         patch("src.gform_filter.requests.get"):  # no actual URL fetch
        result = pipeline.run_pipeline(dry_run=True)

    # Only posts with gform should survive
    growth = result["matches"].get("growth_marketing", [])
    pm = result["matches"].get("product_ai", [])
    assert len(growth) == 1
    assert growth[0]["post_url"].endswith("/growth/1")
    assert len(pm) == 1

    # Dry run must NOT record seen
    assert not (config.CONFIG_DIR / "seen_posts.json").exists()


def test_pipeline_real_run_records_seen_and_delivers(tmp_path):
    _write_prefs(tmp_path)

    def fake_scrape(prefs, queries_by_track):
        return {t: [_normalize_raw(r) for r in _apify_posts(t)] for t in queries_by_track}

    def fake_embed(texts, input_type):
        return np.ones((len(texts), 4), dtype=np.float32)

    def fake_score_one(track, summary, post, client=None):
        return {"score": 8.0, "reason": "good fit"}

    resend_resp = MagicMock()
    resend_resp.status_code = 200
    resend_resp.text = "{}"

    with patch("src.pipeline.sources.scrape_all", side_effect=fake_scrape), \
         patch("src.matcher.embed_texts", side_effect=fake_embed), \
         patch("src.reranker.score_one", side_effect=fake_score_one), \
         patch("src.pipeline.Anthropic"), \
         patch("src.gform_filter.requests.get"), \
         patch("src.delivery.requests.post", return_value=resend_resp):
        result = pipeline.run_pipeline(dry_run=False)

    # Seen file should exist after real run
    seen = json.loads((config.CONFIG_DIR / "seen_posts.json").read_text())
    assert len(seen) >= 2

    # CSV file written
    csvs = list(config.RESULTS_DIR.glob("matches_*.csv"))
    assert csvs

    # Delivery attempted
    assert result["delivery"]["sent"] is True


def test_pipeline_dedup_on_second_run(tmp_path):
    _write_prefs(tmp_path)

    def fake_scrape(prefs, queries_by_track):
        return {t: [_normalize_raw(r) for r in _apify_posts(t)] for t in queries_by_track}

    def fake_embed(texts, input_type):
        return np.ones((len(texts), 4), dtype=np.float32)

    def fake_score_one(track, summary, post, client=None):
        return {"score": 8.0, "reason": "good fit"}

    resend_resp = MagicMock(); resend_resp.status_code = 200; resend_resp.text = "{}"

    with patch("src.pipeline.sources.scrape_all", side_effect=fake_scrape), \
         patch("src.matcher.embed_texts", side_effect=fake_embed), \
         patch("src.reranker.score_one", side_effect=fake_score_one), \
         patch("src.pipeline.Anthropic"), \
         patch("src.gform_filter.requests.get"), \
         patch("src.delivery.requests.post", return_value=resend_resp):
        first = pipeline.run_pipeline(dry_run=False)
        second = pipeline.run_pipeline(dry_run=False)

    # All posts marked seen by the first run; second run should produce 0
    assert sum(len(v) for v in first["matches"].values()) > 0
    assert sum(len(v) for v in second["matches"].values()) == 0


def test_pipeline_cross_track_dedup(tmp_path):
    """A post that matches multiple tracks should be emitted under one track only."""
    prefs = {
        "tracks": [
            {"track": "growth_marketing", "active": True, "roles": ["Growth"], "cities": ["Bangalore"], "include_remote": False, "resume_summary": "growth marketer"},
            {"track": "product_ai", "active": True, "roles": ["PM"], "cities": ["Bangalore"], "include_remote": False, "resume_summary": "AI PM"},
        ]
    }
    config.PREFS_PATH.write_text(json.dumps(prefs))

    # Same post returned by both queries
    shared = {
        "postUrl": "https://linkedin.com/shared/1",
        "full_urn": "urn:li:activity:SHARED",
        "text": "Hiring — apply https://forms.gle/SHARED",
        "author": {"name": "X"},
        "comments": [],
    }

    def fake_scrape(prefs, queries_by_track):
        return {t: [_normalize_raw(shared)] for t in queries_by_track}

    def fake_embed(texts, input_type):
        return np.ones((len(texts), 4), dtype=np.float32)

    def fake_score_one(track, summary, post, client=None):
        return {"score": 9.0, "reason": f"fit for {track}"}

    anthropic_mock = MagicMock()
    # assign_best_track → returns text message with chosen track
    msg = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = "growth_marketing"
    msg.content = [block]
    anthropic_mock.return_value.messages.create.return_value = msg

    resend_resp = MagicMock(); resend_resp.status_code = 200; resend_resp.text = "{}"

    with patch("src.pipeline.sources.scrape_all", side_effect=fake_scrape), \
         patch("src.matcher.embed_texts", side_effect=fake_embed), \
         patch("src.reranker.score_one", side_effect=fake_score_one), \
         patch("src.pipeline.Anthropic", anthropic_mock), \
         patch("src.gform_filter.requests.get"), \
         patch("src.delivery.requests.post", return_value=resend_resp):
        result = pipeline.run_pipeline(dry_run=True)

    all_matches = [m for ms in result["matches"].values() for m in ms]
    assert len(all_matches) == 1
    assert result["matches"]["growth_marketing"]  # winner track
    assert not result["matches"]["product_ai"]    # loser track


def test_pipeline_fail_soft_when_apify_errors(tmp_path):
    _write_prefs(tmp_path)

    with patch("src.pipeline.sources.scrape_all", side_effect=RuntimeError("apify down")):
        result = pipeline.run_pipeline(dry_run=True)

    # Pipeline should return with error captured, not crash
    assert result["matches"] == {}
    assert "error" in result


def test_pipeline_no_active_tracks_returns_empty(tmp_path):
    prefs = {
        "tracks": [
            {"track": "off", "active": False, "roles": ["X"], "cities": ["Y"], "include_remote": False, "resume_summary": ""},
        ]
    }
    config.PREFS_PATH.write_text(json.dumps(prefs))
    result = pipeline.run_pipeline(dry_run=True)
    assert result["matches"] == {}


# ---------- Content-hash dedup across tracks ----------
#
# The same role gets syndicated to multiple sources (LinkedIn posts +
# LinkedIn jobs + Naukri) with DIFFERENT URLs. URL-key dedup at stage 6
# can't catch those — content-key dedup at stage 6b does.


def test_content_dedup_collapses_same_role_across_sources():
    """Same (company, title, location) across two sources → keep one (higher score)."""
    input_ = {
        "product_ai": [
            {
                "source": "linkedin_job",
                "post_url": "https://linkedin.com/jobs/view/1",
                "company": "Acme Inc",
                "title": "Senior Product Manager",
                "location": "Bangalore",
                "llm_score": 7.2,
            },
            {
                "source": "naukri",
                "post_url": "https://naukri.com/job/abc",
                "company": "Acme",
                "title": "Product Manager",
                "location": "Bangalore, Karnataka",
                "llm_score": 6.8,  # lower — should be the one dropped
            },
        ]
    }
    out = pipeline._content_dedup_across_tracks(input_)
    assert len(out["product_ai"]) == 1
    assert out["product_ai"][0]["post_url"].endswith("/jobs/view/1")


def test_content_dedup_keeps_different_roles():
    """Different title_head → different cluster → both preserved."""
    input_ = {
        "product_ai": [
            {
                "source": "linkedin_job",
                "post_url": "a",
                "company": "Acme",
                "title": "Product Manager",
                "location": "Bangalore",
                "llm_score": 7.0,
            },
            {
                "source": "linkedin_job",
                "post_url": "b",
                "company": "Acme",
                "title": "Engineering Manager",  # different role
                "location": "Bangalore",
                "llm_score": 7.0,
            },
        ]
    }
    out = pipeline._content_dedup_across_tracks(input_)
    assert len(out["product_ai"]) == 2


def test_content_dedup_seniority_tokens_stripped():
    """Senior / Lead variants of the same role collapse into one cluster."""
    input_ = {
        "product_ai": [
            {
                "source": "naukri",
                "post_url": "a",
                "company": "Acme",
                "title": "Senior Product Manager",
                "location": "Bangalore",
                "llm_score": 6.0,
            },
            {
                "source": "linkedin_job",
                "post_url": "b",
                "company": "Acme",
                "title": "Lead Product Manager",
                "location": "Bangalore",
                "llm_score": 8.0,
            },
        ]
    }
    out = pipeline._content_dedup_across_tracks(input_)
    assert len(out["product_ai"]) == 1
    # Highest-scored wins
    assert out["product_ai"][0]["post_url"] == "b"


def test_content_dedup_bangalore_bengaluru_alias():
    """Bangalore and Bengaluru should be treated as the same location."""
    input_ = {
        "product_ai": [
            {
                "source": "linkedin_job",
                "post_url": "a",
                "company": "Acme",
                "title": "Product Manager",
                "location": "Bangalore",
                "llm_score": 7.0,
            },
            {
                "source": "naukri",
                "post_url": "b",
                "company": "Acme",
                "title": "Product Manager",
                "location": "Bengaluru",
                "llm_score": 6.0,
            },
        ]
    }
    out = pipeline._content_dedup_across_tracks(input_)
    assert len(out["product_ai"]) == 1


def test_content_dedup_source_preference_breaks_score_tie():
    """When scores tie, linkedin_post > linkedin_job > naukri wins."""
    input_ = {
        "product_ai": [
            {
                "source": "naukri",
                "post_url": "n",
                "company": "Acme",
                "title": "Product Manager",
                "location": "Bangalore",
                "llm_score": 7.5,
            },
            {
                "source": "linkedin_job",
                "post_url": "l",
                "company": "Acme",
                "title": "Product Manager",
                "location": "Bangalore",
                "llm_score": 7.5,
            },
        ]
    }
    out = pipeline._content_dedup_across_tracks(input_)
    assert out["product_ai"][0]["post_url"] == "l"  # linkedin_job wins tie


def test_content_dedup_blank_company_and_title_pass_through():
    """Posts with neither company nor title have no computable key — they
    shouldn't collapse against random others sharing the same location."""
    input_ = {
        "product_ai": [
            {
                "source": "linkedin_post",
                "post_url": "x",
                "company": "",
                "title": "",
                "location": "Bangalore",
                "llm_score": 5.0,
                "author_title": "",  # nothing to extract from
            },
            {
                "source": "linkedin_post",
                "post_url": "y",
                "company": "",
                "title": "",
                "location": "Bangalore",
                "llm_score": 6.0,
                "author_title": "",
            },
        ]
    }
    out = pipeline._content_dedup_across_tracks(input_)
    assert len(out["product_ai"]) == 2


def test_content_dedup_preserves_track_assignment():
    """When two tracks each hold a post for the same cluster, the winner keeps
    its own track — it doesn't jump to the other track."""
    input_ = {
        "growth_marketing": [
            {
                "source": "linkedin_job",
                "post_url": "a",
                "company": "Acme",
                "title": "Growth Manager",
                "location": "Bangalore",
                "llm_score": 7.0,
            },
        ],
        "product_ai": [
            {
                "source": "linkedin_job",
                "post_url": "b",
                "company": "Acme",
                "title": "Growth Manager",
                "location": "Bangalore",
                "llm_score": 8.0,  # winner
            },
        ],
    }
    out = pipeline._content_dedup_across_tracks(input_)
    assert out["growth_marketing"] == []
    assert len(out["product_ai"]) == 1
    assert out["product_ai"][0]["post_url"] == "b"


def test_content_dedup_legal_suffix_stripped_on_company():
    """'Acme Technologies Pvt Ltd' should collapse against 'Acme'."""
    input_ = {
        "product_ai": [
            {
                "source": "linkedin_job",
                "post_url": "a",
                "company": "Acme Technologies Pvt Ltd",
                "title": "Product Manager",
                "location": "Bangalore",
                "llm_score": 7.0,
            },
            {
                "source": "naukri",
                "post_url": "b",
                "company": "Acme",
                "title": "Product Manager",
                "location": "Bangalore",
                "llm_score": 6.5,
            },
        ]
    }
    out = pipeline._content_dedup_across_tracks(input_)
    assert len(out["product_ai"]) == 1


# ---------- helpers ----------

def _normalize_raw(raw):
    """Emulate apify_runner._normalize. We pass normalized dicts as-is because
    the mocked apify_runner.run_all returns the already-normalized shape."""
    from src.apify_runner import _normalize
    return _normalize(raw, query="fake")
