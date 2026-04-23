"""Application-method detection: gform + 19 ATS + LinkedIn Easy Apply + mail."""
from __future__ import annotations

from unittest.mock import patch

import pytest
import requests

from src import application_filter


def _post(text: str = "", comments=None, extra=None) -> dict:
    p = {"text": text, "comments": comments or [], "post_url": "u", "author_name": "a"}
    if extra:
        p.update(extra)
    return p


@pytest.fixture(autouse=True)
def _no_network():
    """Every test in this module goes through gform_filter.find_gform_links,
    which may try to fetch shortener landing pages for URLs with short paths.
    We make requests.get raise so gform_filter falls back to the original URL
    string (its `resolve_redirect` catches RequestException). Tests that want
    to assert on specific gform-shortener resolution live in test_gform_filter."""
    with patch(
        "src.gform_filter.requests.get",
        side_effect=requests.RequestException("network disabled in tests"),
    ):
        yield


# ---------- gform branch (delegates to gform_filter) ----------

def test_gform_signal_is_detected():
    p = _post("Apply at https://forms.gle/abc123")
    info = application_filter.detect_apply_methods(p)
    assert "gform" in info["apply_methods"]
    assert info["gform_urls"] == ["https://forms.gle/abc123"]
    assert "https://forms.gle/abc123" in info["apply_urls"]


# ---------- ATS patterns, one test per platform ----------

def _assert_ats(text: str, expected_method: str, url_fragment: str):
    info = application_filter.detect_apply_methods(_post(text))
    assert expected_method in info["apply_methods"], f"methods={info['apply_methods']}"
    assert any(url_fragment in u for u in info["apply_urls"])


def test_ats_lever():
    _assert_ats("Apply: https://jobs.lever.co/acme/xyz", "lever", "jobs.lever.co/acme")


def test_ats_greenhouse_boards():
    _assert_ats(
        "Apply: https://boards.greenhouse.io/acme/jobs/123",
        "greenhouse",
        "boards.greenhouse.io",
    )


def test_ats_greenhouse_new_job_boards():
    _assert_ats(
        "Apply: https://job-boards.greenhouse.io/company/jobs/456",
        "greenhouse",
        "job-boards.greenhouse.io",
    )


def test_ats_ashby():
    _assert_ats("Apply: https://jobs.ashbyhq.com/acme/xyz", "ashby", "jobs.ashbyhq.com")


def test_ats_workday():
    _assert_ats(
        "Apply: https://acme.wd5.myworkdayjobs.com/External/job/123",
        "workday",
        "myworkdayjobs.com",
    )


def test_ats_workable():
    _assert_ats(
        "Apply: https://apply.workable.com/acme/j/ABC/",
        "workable",
        "apply.workable.com",
    )


def test_ats_smartrecruiters():
    _assert_ats(
        "Apply: https://jobs.smartrecruiters.com/acme/1234",
        "smartrecruiters",
        "smartrecruiters.com",
    )


def test_ats_recruitee():
    _assert_ats(
        "Apply: https://acme.recruitee.com/o/marketing-manager",
        "recruitee",
        "acme.recruitee.com",
    )


def test_ats_teamtailor():
    _assert_ats(
        "Apply: https://acme.teamtailor.com/jobs/123",
        "teamtailor",
        "acme.teamtailor.com",
    )


def test_ats_jobvite():
    _assert_ats(
        "Apply: https://jobs.jobvite.com/acme/job/XYZ",
        "jobvite",
        "jobs.jobvite.com",
    )


def test_ats_icims():
    _assert_ats(
        "Apply: https://careers-acme.icims.com/jobs/123/view",
        "icims",
        "icims.com",
    )


def test_ats_bamboohr():
    _assert_ats(
        "Apply: https://acme.bamboohr.com/careers/42",
        "bamboohr",
        "bamboohr.com",
    )


def test_ats_keka():
    _assert_ats("Apply: https://acme.keka.com/careers/job/123", "keka", "keka.com")


def test_ats_zoho():
    _assert_ats(
        "Apply: https://acme.zohorecruit.in/jobs/Careers/123/Role",
        "zoho",
        "zohorecruit.in",
    )


def test_ats_wellfound():
    _assert_ats(
        "Apply: https://wellfound.com/jobs/123-growth-marketer",
        "wellfound",
        "wellfound.com",
    )


def test_ats_breezy():
    _assert_ats(
        "Apply: https://acme.breezy.hr/p/abc-role",
        "breezy",
        "breezy.hr",
    )


def test_ats_dover():
    _assert_ats(
        "Apply: https://app.dover.com/jobs/abc-123",
        "dover",
        "dover.com",
    )


def test_ats_naukri():
    _assert_ats(
        "Apply: https://www.naukri.com/job-listings-manager-acme-123",
        "naukri",
        "naukri.com",
    )


def test_ats_instahyre():
    _assert_ats(
        "Apply: https://www.instahyre.com/job/1234-growth-manager-acme/",
        "instahyre",
        "instahyre.com",
    )


def test_ats_cutshort():
    _assert_ats(
        "Apply: https://cutshort.io/job/growth-manager-12345",
        "cutshort",
        "cutshort.io",
    )


# ---------- LinkedIn Easy Apply ----------

def test_linkedin_jobs_view_url_triggers_easy_apply():
    p = _post("Hiring! Apply: https://www.linkedin.com/jobs/view/1234567890")
    info = application_filter.detect_apply_methods(p)
    assert "linkedin_easy_apply" in info["apply_methods"]
    assert any("/jobs/view/1234567890" in u for u in info["apply_urls"])


def test_easy_apply_phrase_without_url():
    p = _post("We're hiring. Easy Apply below!")
    info = application_filter.detect_apply_methods(p)
    assert "linkedin_easy_apply" in info["apply_methods"]


def test_apply_on_linkedin_phrase():
    p = _post("Senior PM role. Apply on LinkedIn.")
    info = application_filter.detect_apply_methods(p)
    assert "linkedin_easy_apply" in info["apply_methods"]


def test_no_easy_apply_without_signals():
    p = _post("Senior PM role. Great team.")
    info = application_filter.detect_apply_methods(p)
    assert "linkedin_easy_apply" not in info["apply_methods"]


# ---------- Mail-to-apply ----------

def test_mail_signal_requires_both_email_and_intent():
    # Both present → detected
    p = _post("Share your resume at hiring@acme.com")
    info = application_filter.detect_apply_methods(p)
    assert "mail" in info["apply_methods"]
    assert "hiring@acme.com" in info["apply_emails"]


def test_mail_signal_skipped_when_no_intent_phrase():
    # Random email without intent phrase → NOT detected
    p = _post("Check our blog: john@acme.com wrote about it")
    info = application_filter.detect_apply_methods(p)
    assert "mail" not in info["apply_methods"]
    assert info["apply_emails"] == []


def test_mail_signal_skipped_when_intent_without_email():
    # "Send your CV" but no email → NOT detected
    p = _post("We're hiring! Send your CV to our recruiter.")
    info = application_filter.detect_apply_methods(p)
    assert "mail" not in info["apply_methods"]


def test_mail_phrase_drop_a_mail():
    p = _post("Drop a mail at careers@acme.com")
    info = application_filter.detect_apply_methods(p)
    assert "mail" in info["apply_methods"]
    assert "careers@acme.com" in info["apply_emails"]


def test_mail_phrase_email_us_at():
    p = _post("Email us at jobs@acme.com for the role.")
    info = application_filter.detect_apply_methods(p)
    assert "mail" in info["apply_methods"]


def test_mail_phrase_apply_via_email():
    p = _post("Apply via email: talent@acme.com")
    info = application_filter.detect_apply_methods(p)
    assert "mail" in info["apply_methods"]


def test_mail_phrase_share_your_profile():
    p = _post("Interested? Share your profile at hiring@acme.com")
    info = application_filter.detect_apply_methods(p)
    assert "mail" in info["apply_methods"]


def test_mail_noreply_address_excluded():
    p = _post("Send your CV to noreply@acme.com")
    info = application_filter.detect_apply_methods(p)
    # Intent is present, but the only email is blocklisted → nothing left
    assert "mail" not in info["apply_methods"]
    assert info["apply_emails"] == []


def test_mail_signal_picks_email_from_comment():
    p = _post(
        "Hiring Growth Marketer!",
        comments=[{"text": "Send your CV to recruit@acme.com", "replies": []}],
    )
    info = application_filter.detect_apply_methods(p)
    assert "mail" in info["apply_methods"]
    assert "recruit@acme.com" in info["apply_emails"]


# ---------- combined signals ----------

def test_multiple_methods_on_same_post():
    p = _post(
        "Hiring! Apply https://jobs.lever.co/acme/xyz OR email us at hr@acme.com"
    )
    info = application_filter.detect_apply_methods(p)
    assert "lever" in info["apply_methods"]
    assert "mail" in info["apply_methods"]


def test_no_signals_means_no_methods():
    p = _post("Hiring Growth Marketer, DM me.")
    info = application_filter.detect_apply_methods(p)
    assert info["apply_methods"] == []
    assert info["apply_urls"] == []


# ---------- filter_posts_with_apply_method ----------

def test_filter_keeps_only_posts_with_method():
    posts = [
        _post("no signal"),
        _post("https://forms.gle/YES"),
        _post("https://jobs.lever.co/acme/xyz"),
        _post("dm me to apply"),  # no actionable signal
        _post("Send CV at jobs@acme.com"),
    ]
    kept = application_filter.filter_posts_with_apply_method(posts)
    assert len(kept) == 3
    methods = [k["apply_methods"] for k in kept]
    assert ["gform"] in methods
    assert ["lever"] in methods
    assert ["mail"] in methods


def test_filter_empty_input():
    assert application_filter.filter_posts_with_apply_method([]) == []


def test_filter_preserves_original_fields():
    posts = [_post("https://forms.gle/XYZ", extra={"post_url": "http://linkedin.com/1"})]
    kept = application_filter.filter_posts_with_apply_method(posts)
    assert kept[0]["post_url"] == "http://linkedin.com/1"
    assert kept[0]["apply_methods"] == ["gform"]


def test_method_distribution_counts_per_method():
    posts = [
        {"apply_methods": ["gform"]},
        {"apply_methods": ["gform", "mail"]},
        {"apply_methods": ["lever"]},
        {"apply_methods": []},
    ]
    dist = application_filter.method_distribution(posts)
    assert dist == {"gform": 2, "mail": 1, "lever": 1}


# ---------- URL trimming ----------

def test_ats_url_trailing_punctuation_trimmed():
    p = _post("Apply: (https://jobs.lever.co/acme/xyz).")
    info = application_filter.detect_apply_methods(p)
    assert any(u == "https://jobs.lever.co/acme/xyz" for u in info["apply_urls"])


def test_detect_returns_all_four_keys_always():
    """Regardless of signals, the shape must be stable."""
    p = _post("completely empty of signals")
    info = application_filter.detect_apply_methods(p)
    assert set(info.keys()) == {"apply_methods", "apply_urls", "apply_emails", "gform_urls"}


# ---------- structured-source shortcut ----------
# Job-board records arrive with a pre-known apply URL — we should treat the
# listing itself as the apply channel and skip the free-text scan entirely.


def test_structured_linkedin_job_passes_without_scanning_body():
    """A linkedin_job record has an apply_url up front — no #hiring phrase needed."""
    p = {
        "source": "linkedin_job",
        "post_url": "https://linkedin.com/jobs/view/9",
        "apply_url": "https://acme.com/apply/9",
        "text": "A plain JD with no apply phrasing whatsoever.",
        "comments": [],
    }
    info = application_filter.detect_apply_methods(p)
    assert info["apply_methods"] == ["linkedin_job"]
    assert info["apply_urls"] == ["https://acme.com/apply/9"]
    # Free-text scan must NOT have fired for mail/gform etc.
    assert info["apply_emails"] == []
    assert info["gform_urls"] == []


def test_structured_naukri_job_passes_with_listing_as_apply():
    """Naukri's listing URL IS the apply target — if no apply_url, use post_url."""
    p = {
        "source": "naukri",
        "post_url": "https://naukri.com/job/42",
        "apply_url": None,  # Naukri often leaves this blank
        "text": "Some JD",
        "comments": [],
    }
    info = application_filter.detect_apply_methods(p)
    assert info["apply_methods"] == ["naukri"]
    assert info["apply_urls"] == ["https://naukri.com/job/42"]


def test_structured_source_without_any_url_returns_method_but_no_urls():
    """Degenerate record — shouldn't crash, just emits empty URL list."""
    p = {"source": "linkedin_job", "post_url": "", "apply_url": None, "text": "JD"}
    info = application_filter.detect_apply_methods(p)
    assert info["apply_methods"] == ["linkedin_job"]
    assert info["apply_urls"] == []


def test_linkedin_post_source_still_requires_free_text_signal():
    """The shortcut applies ONLY to structured sources — linkedin_post still
    needs a real apply phrase in the body."""
    p = {
        "source": "linkedin_post",
        "post_url": "https://linkedin.com/p/1",
        "text": "Random recruiter rant, no apply method.",
        "comments": [],
    }
    info = application_filter.detect_apply_methods(p)
    assert info["apply_methods"] == []


# ---------- Soft-apply (DM / comment-interested) channel ----------
#
# Two-gated: soft-apply language AND (hiring-intent author OR recruiter-style
# body). The gate keeps precision high — a post that just says "DM me your
# thoughts" shouldn't fire; but "DM me your resume — we're hiring 3 PMs"
# should.


def test_soft_dm_fires_for_dm_me_plus_recruiter_title():
    """DM-me language + author is a recruiter → soft_dm fires."""
    p = {
        "source": "linkedin_post",
        "text": "Hiring ops analyst. DM me your profile.",
        "author_title": "Talent Acquisition Partner at Razorpay",
        "comments": [],
    }
    info = application_filter.detect_apply_methods(p)
    assert "soft_dm" in info["apply_methods"]


def test_soft_dm_fires_for_comment_interested_plus_recruiter_language():
    """Comment-interested language + recruiter-style body → fires even if
    author title isn't in the recruiter archetype list."""
    p = {
        "source": "linkedin_post",
        "text": "We're hiring 3 Product Managers at Leap. Comment 'interested' below.",
        "author_title": "Senior Growth Lead at Leap",
        "comments": [],
    }
    info = application_filter.detect_apply_methods(p)
    assert "soft_dm" in info["apply_methods"]


def test_soft_dm_does_not_fire_without_hiring_intent():
    """DM-language without any hiring-intent signal is just generic engagement
    bait — should NOT fire."""
    p = {
        "source": "linkedin_post",
        "text": "DM me your thoughts on the new AI bill.",
        "author_title": "Policy Analyst",
        "comments": [],
    }
    info = application_filter.detect_apply_methods(p)
    assert "soft_dm" not in info["apply_methods"]


def test_soft_dm_does_not_fire_without_dm_language():
    """Hiring-intent without any soft-apply phrase is a recruiter-style post
    but has no clear apply channel — our other gates (ATS/mail/gform) should
    catch it if there's a real URL; soft_dm only fires on DM-ish phrasing."""
    p = {
        "source": "linkedin_post",
        "text": "We're hiring 3 PMs at Leap. Great team, great equity.",
        "author_title": "Talent Lead at Leap",
        "comments": [],
    }
    info = application_filter.detect_apply_methods(p)
    assert "soft_dm" not in info["apply_methods"]


def test_soft_dm_scoped_to_linkedin_post_only():
    """Job-board sources never hit this path — they have structured apply_url."""
    p = {
        "source": "linkedin_job",
        "post_url": "https://linkedin.com/jobs/view/9",
        "apply_url": "https://acme.com/apply/9",
        "text": "DM me your resume. We're hiring 3 PMs.",
        "author_title": "Recruiter",
        "comments": [],
    }
    info = application_filter.detect_apply_methods(p)
    # structured-source shortcut returns [source] only — soft_dm must not sneak in
    assert "soft_dm" not in info["apply_methods"]


def test_soft_dm_flag_disables_signal(monkeypatch):
    """ENABLE_SOFT_APPLY=false → the signal never fires, even on perfect matches."""
    from src import config

    monkeypatch.setattr(config, "ENABLE_SOFT_APPLY", False)
    p = {
        "source": "linkedin_post",
        "text": "DM me your resume. Hiring 3 PMs at Leap.",
        "author_title": "Talent Lead at Leap",
        "comments": [],
    }
    info = application_filter.detect_apply_methods(p)
    assert "soft_dm" not in info["apply_methods"]


def test_soft_dm_comment_walk_catches_dm_phrase_in_comments():
    """Some recruiters drop the DM ask as a self-reply in comments after the
    original post. We walk comments just like the other channels do."""
    p = {
        "source": "linkedin_post",
        "text": "Hiring for growth roles. We're expanding the team.",
        "author_title": "Head of Talent at Razorpay",
        "comments": [
            {"text": "Details? I'm interested.", "author": "Someone"},
            {"text": "Sure — DM me your CV for the JD.", "author": "Op"},
        ],
    }
    info = application_filter.detect_apply_methods(p)
    assert "soft_dm" in info["apply_methods"]


def test_soft_dm_passes_filter_posts_with_apply_method():
    """End-to-end: a post with ONLY soft-apply signal still passes the
    top-level filter_posts_with_apply_method gate (previously it was
    hard-rejected for having no gform/ATS/mail)."""
    p = {
        "source": "linkedin_post",
        "post_url": "https://linkedin.com/posts/1",
        "text": "Hiring 3 PMs at Leap. DM me your resume.",
        "author_title": "Talent Lead at Leap",
        "comments": [],
    }
    kept = application_filter.filter_posts_with_apply_method([p])
    assert len(kept) == 1
    assert "soft_dm" in kept[0]["apply_methods"]
