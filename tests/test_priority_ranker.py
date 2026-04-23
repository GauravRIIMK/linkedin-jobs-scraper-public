"""Priority boost layer: source, company, remote, location, apply-in-comments.

Boosts are applied AFTER Haiku reranking and BEFORE email render. Each signal
is configurable via config.PRIORITY_BOOST_*; this test pins the default
behavior so accidental regressions surface loudly.
"""
from __future__ import annotations

import json

import pytest

from src import config, priority_ranker


def _write_tier1(tmp_companies):
    """Write a scoped tier-1 list and force the module to reload."""
    config.INDIAN_TIER1_COMPANIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.INDIAN_TIER1_COMPANIES_PATH.write_text(
        json.dumps({"companies": tmp_companies}), encoding="utf-8"
    )
    priority_ranker.reload_tier1_companies()


# ---------------------------- Source boost ----------------------------

def test_linkedin_post_gets_source_boost():
    post = {"source": "linkedin_post", "llm_score": 7.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert boosted[0]["priority_boost"] >= config.PRIORITY_BOOST_LINKEDIN_POST
    assert "LinkedIn post" in boosted[0]["priority_reasons"]


def test_linkedin_job_no_source_boost():
    post = {"source": "linkedin_job", "llm_score": 7.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "LinkedIn post" not in boosted[0]["priority_reasons"]


def test_naukri_no_source_boost():
    post = {"source": "naukri", "llm_score": 7.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert boosted[0]["priority_boost"] == 0.0
    assert boosted[0]["priority_reasons"] == []


# ---------------------------- Apply-in-comments boost ----------------------------

def test_apply_in_comments_boost_fires_for_linkedin_post():
    url = "https://forms.gle/ABC"
    post = {
        "source": "linkedin_post",
        "llm_score": 7.0,
        "apply_urls": [url],
        "comments": [{"text": f"Apply here: {url}"}],
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    reasons = boosted[0]["priority_reasons"]
    assert "Apply in comments" in reasons


def test_apply_in_comments_not_fired_when_link_only_in_body():
    url = "https://forms.gle/ABC"
    post = {
        "source": "linkedin_post",
        "llm_score": 7.0,
        "text": f"Hiring! Apply here: {url}",
        "apply_urls": [url],
        "comments": [],
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Apply in comments" not in boosted[0]["priority_reasons"]


def test_apply_in_comments_walks_nested_replies():
    url = "https://boards.greenhouse.io/abc/jobs/123"
    post = {
        "source": "linkedin_post",
        "llm_score": 7.0,
        "apply_urls": [url],
        "comments": [
            {
                "text": "Great opportunity!",
                "replies": [{"text": f"Link: {url}"}],
            }
        ],
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Apply in comments" in boosted[0]["priority_reasons"]


def test_apply_in_comments_for_email_in_comments():
    """If the apply channel is an email (mail-to-apply), check comments for the
    email address too."""
    post = {
        "source": "linkedin_post",
        "llm_score": 7.0,
        "apply_emails": ["careers@acme.com"],
        "comments": [{"text": "Send CV to careers@acme.com"}],
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Apply in comments" in boosted[0]["priority_reasons"]


def test_apply_in_comments_skips_for_job_board_source():
    url = "https://linkedin.com/jobs/view/123"
    post = {
        "source": "linkedin_job",
        "llm_score": 7.0,
        "apply_urls": [url],
        "comments": [{"text": f"Link: {url}"}],  # wouldn't exist for job board but test anyway
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Apply in comments" not in boosted[0]["priority_reasons"]


# ---------------------------- OP-authored comment (recruiter self-reply) ----------------------------

def test_apply_in_comments_op_selfreply_fires_extra_boost():
    """When the apply URL is dropped in a comment authored BY THE POST AUTHOR,
    that's the recruiter self-reply pattern — should earn 'Recruiter-dropped link'
    on top of the generic 'Apply in comments' reason."""
    url = "https://forms.gle/XYZ"
    post = {
        "source": "linkedin_post",
        "llm_score": 6.0,
        "author_name": "Priya Nair",
        "apply_urls": [url],
        "comments": [
            {"text": f"Apply here: {url}", "author": "Priya Nair"},
        ],
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    reasons = boosted[0]["priority_reasons"]
    assert "Apply in comments" in reasons
    assert "Recruiter-dropped link" in reasons


def test_apply_in_comments_random_commenter_does_not_fire_op_boost():
    """A random commenter posting the form should still count as 'Apply in
    comments' but NOT 'Recruiter-dropped link'."""
    url = "https://forms.gle/XYZ"
    post = {
        "source": "linkedin_post",
        "llm_score": 6.0,
        "author_name": "Priya Nair",
        "apply_urls": [url],
        "comments": [
            {"text": f"Interesting! {url}", "author": "Rahul Kumar"},
        ],
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    reasons = boosted[0]["priority_reasons"]
    assert "Apply in comments" in reasons
    assert "Recruiter-dropped link" not in reasons


def test_op_selfreply_in_nested_reply_still_detected():
    """OP may reply to a question in their own thread — nested replies count."""
    url = "https://boards.greenhouse.io/acme/jobs/1"
    post = {
        "source": "linkedin_post",
        "llm_score": 6.0,
        "author_name": "Priya Nair",
        "apply_urls": [url],
        "comments": [
            {
                "text": "What's the link?",
                "author": "Someone Else",
                "replies": [{"text": f"Here you go: {url}", "author": "Priya Nair"}],
            }
        ],
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    reasons = boosted[0]["priority_reasons"]
    assert "Apply in comments" in reasons
    assert "Recruiter-dropped link" in reasons


def test_op_selfreply_case_insensitive_author_match():
    """Author names on LinkedIn can have casing differences between the post
    payload and the comment payload — match should be case-insensitive."""
    url = "https://forms.gle/XYZ"
    post = {
        "source": "linkedin_post",
        "llm_score": 6.0,
        "author_name": "Priya Nair",
        "apply_urls": [url],
        "comments": [
            {"text": f"Link: {url}", "author": "PRIYA NAIR"},
        ],
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Recruiter-dropped link" in boosted[0]["priority_reasons"]


# ---------------------------- Hiring-intent author ----------------------------

@pytest.mark.parametrize(
    "title",
    [
        "Talent Acquisition Partner at Razorpay",
        "Senior Recruiter",
        "Head of Talent",
        "HR Business Partner",
        "HRBP - Product Engineering",
        "Hiring Manager",
        "Co-Founder & CEO",
        "Founder",
        "CTO at Acme",
        "Chief of Staff",
        "People Operations Lead",
        "Chief People Officer",
    ],
)
def test_hiring_intent_author_fires(title):
    post = {"source": "linkedin_post", "author_title": title, "llm_score": 5.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Hiring manager / recruiter" in boosted[0]["priority_reasons"]


@pytest.mark.parametrize(
    "title",
    [
        "Software Engineer",
        "Data Scientist",
        "Product Manager",
        "UX Designer",
        "Growth Marketer",
        "",  # empty title — no match
    ],
)
def test_hiring_intent_author_doesnt_fire_for_non_hirer(title):
    post = {"source": "linkedin_post", "author_title": title, "llm_score": 5.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Hiring manager / recruiter" not in boosted[0]["priority_reasons"]


def test_hiring_intent_skipped_for_job_board_source():
    """author_title is meaningful only on linkedin_post; don't try to parse
    it on structured job-board records (Naukri/LinkedIn jobs)."""
    post = {"source": "linkedin_job", "author_title": "Recruiter", "llm_score": 5.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Hiring manager / recruiter" not in boosted[0]["priority_reasons"]


# ---------------------------- Indian tier-1 company boost ----------------------------

def test_indian_tier1_exact_match_on_company_field():
    _write_tier1(["Razorpay", "Zerodha"])
    post = {"company": "Razorpay", "llm_score": 6.0, "source": "linkedin_job"}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Tier-1 Indian company" in boosted[0]["priority_reasons"]


def test_indian_tier1_substring_in_structured_field():
    _write_tier1(["Razorpay"])
    post = {"company": "Razorpay Software Private Limited", "llm_score": 6.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Tier-1 Indian company" in boosted[0]["priority_reasons"]


def test_indian_tier1_body_scan_in_head_only():
    _write_tier1(["Zerodha"])
    post = {
        "llm_score": 5.0,
        "jd_text": "Join Zerodha's growth team. We are hiring for ...",
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Tier-1 Indian company" in boosted[0]["priority_reasons"]


def test_indian_tier1_no_match_when_company_deep_in_body():
    _write_tier1(["Flipkart"])
    post = {
        "llm_score": 5.0,
        "jd_text": ("Hiring for ... " * 200) + "Flipkart",  # mention is beyond head scan
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Tier-1 Indian company" not in boosted[0]["priority_reasons"]


def test_short_company_name_doesnt_false_positive_in_body():
    _write_tier1(["CRED", "Jar"])  # 4-char names but the body-scan path skips <5 chars
    # "Jar" is <5 chars → shouldn't match body text "cookie jar"
    post = {"llm_score": 5.0, "jd_text": "We keep cookies in a jar. Join us!"}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Tier-1 Indian company" not in boosted[0]["priority_reasons"]


# ---------------------------- Remote boost ----------------------------

@pytest.mark.parametrize(
    "loc,body,expected",
    [
        ("Remote, India", "", True),
        ("", "This is a fully remote role.", True),
        ("", "100% remote position.", True),
        ("", "Work from home (WFH)", True),
        ("", "Anywhere in India.", True),
        ("", "Remote-first team.", True),
        ("Bangalore", "", False),
        ("", "We have a remote chance of success.", False),  # no match — not a remote-role phrase
    ],
)
def test_remote_detection(loc, body, expected):
    post = {"location": loc, "jd_text": body, "llm_score": 5.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert ("Remote" in boosted[0]["priority_reasons"]) is expected


# ---------------------------- User city boost ----------------------------

def test_user_city_boost(monkeypatch):
    monkeypatch.setattr(config, "USER_CITIES", ["gurgaon", "delhi"])
    post = {"location": "Gurgaon, Haryana", "llm_score": 5.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Near you" in boosted[0]["priority_reasons"]


def test_user_city_no_match(monkeypatch):
    monkeypatch.setattr(config, "USER_CITIES", ["gurgaon"])
    post = {"location": "Chennai", "llm_score": 5.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Near you" not in boosted[0]["priority_reasons"]


# ---------------------------- India-location boost ----------------------------

@pytest.mark.parametrize(
    "location",
    [
        "India",
        "Bengaluru",
        "Bangalore, India",
        "Hyderabad, Telangana",
        "Chennai, Tamil Nadu",
        "Pune",
        "Kerala",
        "Ahmedabad",
        "Delhi NCR",
        "Mumbai, Maharashtra",
    ],
)
def test_india_boost_fires_for_indian_locations(monkeypatch, location):
    # Scope USER_CITIES tight so the user-city path doesn't short-circuit.
    monkeypatch.setattr(config, "USER_CITIES", ["gurgaon"])
    _write_tier1([])  # no tier-1 match
    post = {"location": location, "llm_score": 5.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "India" in boosted[0]["priority_reasons"], (
        f"expected 'India' badge for {location!r}, got {boosted[0]['priority_reasons']}"
    )


def test_india_boost_suppressed_when_tier1_fires(monkeypatch):
    """Tier-1 is the strongest Indian signal; stacking India on top would
    double-count the same physical cue."""
    monkeypatch.setattr(config, "USER_CITIES", [])
    _write_tier1(["Zerodha"])
    post = {"company": "Zerodha", "location": "Bangalore, India", "llm_score": 5.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Tier-1 Indian company" in boosted[0]["priority_reasons"]
    assert "India" not in boosted[0]["priority_reasons"]


def test_india_boost_suppressed_when_user_city_fires(monkeypatch):
    monkeypatch.setattr(config, "USER_CITIES", ["gurgaon"])
    _write_tier1([])
    post = {"location": "Gurgaon, Haryana", "llm_score": 5.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Near you" in boosted[0]["priority_reasons"]
    assert "India" not in boosted[0]["priority_reasons"]


@pytest.mark.parametrize(
    "location",
    [
        "San Francisco, CA",
        "New York, United States",
        "London, UK",
        "Singapore",
        "Dubai",
        "Sydney",
        "Berlin",
        "Toronto",
    ],
)
def test_not_in_india_penalty_for_foreign_locations(monkeypatch, location):
    monkeypatch.setattr(config, "USER_CITIES", ["gurgaon"])
    _write_tier1([])
    post = {"location": location, "llm_score": 5.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Not in India" in boosted[0]["priority_reasons"]
    # Penalty should push the score below the base.
    assert boosted[0]["priority_boost"] < 0


def test_not_in_india_not_applied_when_india_detected(monkeypatch):
    monkeypatch.setattr(config, "USER_CITIES", [])
    _write_tier1([])
    post = {"location": "Remote · India", "llm_score": 5.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Not in India" not in boosted[0]["priority_reasons"]


def test_india_body_detection_with_preamble(monkeypatch):
    """When structured location is missing, the body preamble 'based in
    Bengaluru' should still geolocate the role."""
    monkeypatch.setattr(config, "USER_CITIES", [])
    _write_tier1([])
    post = {
        "jd_text": "Exciting opportunity! Based in Bengaluru, this role joins our growth team.",
        "llm_score": 5.0,
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "India" in boosted[0]["priority_reasons"]


def test_india_body_casual_mention_does_not_fire(monkeypatch):
    """A casual 'we have customers in India' in body without a preamble must
    NOT fire the India boost — too noisy."""
    monkeypatch.setattr(config, "USER_CITIES", [])
    _write_tier1([])
    post = {
        "jd_text": "Global SaaS company with customers in India, US, and EU. Role is based in San Francisco.",
        "location": "San Francisco",
        "llm_score": 5.0,
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    # Structured location names SF → "Not in India" fires, India does not.
    assert "India" not in boosted[0]["priority_reasons"]
    assert "Not in India" in boosted[0]["priority_reasons"]


# ---------------------------- Compounded boost + clamp ----------------------------

def test_score_clamped_at_10():
    _write_tier1(["Zerodha"])
    post = {
        "company": "Zerodha",
        "source": "linkedin_post",
        "location": "Remote, India",
        "llm_score": 9.5,  # + lots of boost → would exceed 10
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    assert boosted[0]["llm_score"] <= 10.0
    assert boosted[0]["base_llm_score"] == 9.5


def test_multiple_boosts_stack(monkeypatch):
    _write_tier1(["Zerodha"])
    monkeypatch.setattr(config, "USER_CITIES", ["bangalore"])
    post = {
        "company": "Zerodha",
        "source": "linkedin_post",
        "location": "Bangalore",
        "llm_score": 6.0,
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    reasons = boosted[0]["priority_reasons"]
    assert "Tier-1 Indian company" in reasons
    assert "LinkedIn post" in reasons
    assert "Near you" in reasons
    # Boost = 1.0 + 0.6 + 0.3 = 1.9 by defaults
    assert boosted[0]["priority_boost"] > 1.0


# ---------------------------- Sort + preserve keys ----------------------------

def test_apply_boosts_sorts_descending():
    """Higher boosted score should come first after applying boosts."""
    posts = [
        {"id": "A", "llm_score": 7.0, "source": "naukri"},               # no boost → 7.0
        {"id": "B", "llm_score": 6.5, "source": "linkedin_post"},        # +0.6 → 7.1
    ]
    boosted = priority_ranker.apply_priority_boosts(posts)
    ids = [p["id"] for p in boosted]
    assert ids[0] == "B"  # now ranked higher
    assert ids[1] == "A"


def test_apply_boosts_preserves_existing_keys():
    post = {
        "id": "x",
        "llm_score": 6.0,
        "llm_reason": "fits",
        "apply_urls": ["https://a.com"],
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    for k in ("id", "llm_reason", "apply_urls"):
        assert boosted[0][k] == post[k]
    # New fields added:
    for k in ("base_llm_score", "priority_boost", "priority_reasons"):
        assert k in boosted[0]


# ---------------------------- Recruiter-style language ----------------------------
#
# Direct regression tests for every archetype the user called out. If any of
# these stops firing, a real-world recruiter post will silently lose its boost.
# Each test is anchored to a specific pattern so the failure message points at
# the exact phrasing that broke.

def _post_with_body(text: str, **extra) -> dict:
    """Helper: build a minimal linkedin_post dict with the given body text."""
    return {
        "source": "linkedin_post",
        "text": text,
        "llm_score": 7.0,
        **extra,
    }


# -- archetype 1: "We're hiring N people at <Company>" ---------------------
def test_recruiter_language_were_hiring_n_at_company():
    p = _post_with_body("We're hiring 25 people at Cars24 to build our next-gen marketplace.")
    assert priority_ranker._has_recruiter_language(p) is True


def test_recruiter_language_we_are_hiring():
    p = _post_with_body("We are hiring engineers across multiple teams this quarter.")
    assert priority_ranker._has_recruiter_language(p) is True


# -- archetype 2: role hashtag cluster (#categorymanagement #ecommerce) ----
def test_recruiter_language_hashtag_cluster_fires():
    # Hashtag-only post: needs ≥2 hiring hashtags to fire
    p = _post_with_body(
        "Exciting role details in the comments! #hiring #nowhiring #categorymanagement"
    )
    assert priority_ranker._has_recruiter_language(p) is True


def test_recruiter_language_single_hashtag_does_not_fire():
    # One #hiring alone is ambiguous (could be a reshare) — require 2+
    p = _post_with_body("Interesting read on the job market. #hiring")
    assert priority_ranker._has_recruiter_language(p) is False


# -- archetype 3: bare body (feed update) — covered by phrase patterns -----
def test_recruiter_language_naked_hiring_announcement():
    p = _post_with_body(
        "Our team has grown 3x this year and we need an Associate PM to help us "
        "scale further. Apply now via the link in bio."
    )
    assert priority_ranker._has_recruiter_language(p) is True


# -- archetype 4: "Looking for a/an <role> who…" ---------------------------
def test_recruiter_language_looking_for_a_role():
    p = _post_with_body(
        "Looking for an Associate Chief of Staff who wants to work directly "
        "with the founder on 0->1 initiatives."
    )
    assert priority_ranker._has_recruiter_language(p) is True


# -- archetype 5: "We're hiring N <Role>s at <Company>" --------------------
def test_recruiter_language_were_hiring_specific_count_role():
    p = _post_with_body(
        "We're hiring 3 Product Managers at Leap — come build the future of "
        "international education with us!"
    )
    assert priority_ranker._has_recruiter_language(p) is True


# -- archetype 6: "<Company>'s hiring <Role>" (possessive) -----------------
def test_recruiter_language_company_possessive_hiring():
    p = _post_with_body(
        "Trupeer's hiring Content Marketer, Product Managers, and Growth leads. "
        "DM or apply below."
    )
    assert priority_ranker._has_recruiter_language(p) is True


def test_recruiter_language_company_is_hiring():
    p = _post_with_body("Razorpay is hiring across multiple engineering roles.")
    assert priority_ranker._has_recruiter_language(p) is True


# -- archetype 8: "At <Company> we are looking for a <role>" ---------------
def test_recruiter_language_at_company_we_are_looking():
    p = _post_with_body(
        "At Scaler we are looking for a Strategy Associate to join the "
        "founder's office. Full JD in the first comment."
    )
    assert priority_ranker._has_recruiter_language(p) is True


# -- archetype 9: pure hashtag cluster with domain tags --------------------
def test_recruiter_language_hashtag_hiring_domain_cluster():
    p = _post_with_body("Opening for an AI engineer. #hiring #AIjobs #automation")
    # #hiring plus at least one more hiring hashtag → fires
    # (here #AIjobs isn't in the hashtag list, but #hiring alone should not,
    # so verify the phrase trigger saves this via "Opening for an …")
    assert priority_ranker._has_recruiter_language(p) is True


# -- looking to connect / looking to hire ----------------------------------
def test_recruiter_language_looking_to_connect_with_professionals():
    p = _post_with_body(
        "Looking to connect with professionals with 4-6 years of experience "
        "in category management for a leadership role."
    )
    assert priority_ranker._has_recruiter_language(p) is True


def test_recruiter_language_looking_to_hire():
    p = _post_with_body("We're looking to hire a senior backend engineer this month.")
    assert priority_ranker._has_recruiter_language(p) is True


# -- CTA phrases -----------------------------------------------------------
def test_recruiter_language_apply_now_cta():
    p = _post_with_body("New role opened today. Apply now via the link below.")
    assert priority_ranker._has_recruiter_language(p) is True


def test_recruiter_language_join_our_team():
    p = _post_with_body("Come join our team — we're rebuilding payments infra.")
    assert priority_ranker._has_recruiter_language(p) is True


# -- negative cases (false-positive avoidance) ----------------------------
def test_recruiter_language_hiring_trends_thinkpiece_does_not_fire():
    # Industry commentary — should NOT fire
    p = _post_with_body(
        "Hiring trends in 2026 show a 40% dip in junior engineer roles across "
        "US tech. Curious what the India market looks like."
    )
    assert priority_ranker._has_recruiter_language(p) is False


def test_recruiter_language_got_hired_celebration_does_not_fire():
    # Celebration post — passive "hired", not active "hiring"
    p = _post_with_body(
        "Just got hired at Cars24 as a Senior Product Manager! Grateful to "
        "the team for the opportunity."
    )
    assert priority_ranker._has_recruiter_language(p) is False


def test_recruiter_language_looking_for_feedback_does_not_fire():
    # Info request, not hiring — no indefinite article + role
    p = _post_with_body("Looking for feedback on our new pricing page. Thoughts?")
    assert priority_ranker._has_recruiter_language(p) is False


def test_recruiter_language_source_guard():
    # Job-board sources should not fire even if body contains recruiter language
    # (their machine-generated JDs can have similar phrasing and we don't want
    #  the boost to compound with the structured-source baseline).
    p = {
        "source": "naukri",
        "text": "We are hiring 3 engineers at Infosys.",
        "llm_score": 7.0,
    }
    assert priority_ranker._has_recruiter_language(p) is False


def test_recruiter_language_empty_body_does_not_fire():
    p = _post_with_body("")
    assert priority_ranker._has_recruiter_language(p) is False


# -- end-to-end: boost applied + reason label populated -------------------
def test_recruiter_language_end_to_end_adds_boost_and_reason():
    p = _post_with_body(
        "We're hiring 3 Product Managers at Leap. Apply via the link in comments."
    )
    boosted = priority_ranker.apply_priority_boosts([p])[0]
    assert "Recruiter-style post" in boosted["priority_reasons"]
    assert boosted["priority_boost"] >= config.PRIORITY_BOOST_RECRUITER_LANGUAGE


def test_recruiter_language_stacks_with_author_signal():
    # Both signals fire → additive boost
    p = _post_with_body(
        "At Scaler we are looking for a Strategy Associate. JD in comments.",
        author_title="Chief of Staff @ Scaler",
    )
    boosted = priority_ranker.apply_priority_boosts([p])[0]
    assert "Hiring manager / recruiter" in boosted["priority_reasons"]
    assert "Recruiter-style post" in boosted["priority_reasons"]
    # Both boosts stack (on top of LinkedIn post source boost)
    expected_min = (
        config.PRIORITY_BOOST_LINKEDIN_POST
        + config.PRIORITY_BOOST_HIRING_INTENT
        + config.PRIORITY_BOOST_RECRUITER_LANGUAGE
    )
    assert boosted["priority_boost"] >= expected_min - 0.001


def test_recruiter_language_fires_without_author_title():
    # The critical case the user flagged: IC without "recruiter" title posting
    # about their team's hire. _is_hiring_intent_author() misses this; the
    # new body-language signal must carry it.
    p = _post_with_body(
        "We're hiring 25 people at Cars24 to build our next-gen marketplace.",
        author_title="Senior Growth Lead @ Cars24",  # not a recruiter title
    )
    boosted = priority_ranker.apply_priority_boosts([p])[0]
    assert "Hiring manager / recruiter" not in boosted["priority_reasons"]
    assert "Recruiter-style post" in boosted["priority_reasons"]


# ---------------------------- Weighted hashtag behavior ----------------------------
# Old rule was "≥2 hashtags to fire" — the new weighted system should keep that
# historical behavior for generic tags but ALSO fire on a single strong tag.

def test_nowhiring_single_hashtag_fires():
    # #nowhiring alone is a strong-enough signal (3.0 > threshold 2.5)
    p = _post_with_body("Details in comments. #nowhiring")
    assert priority_ranker.has_recruiter_language(p) is True


def test_hiringalert_single_hashtag_fires():
    # #hiringalert alone is strong enough
    p = _post_with_body("Apply now. #hiringalert")
    assert priority_ranker.has_recruiter_language(p) is True


def test_plain_hiring_single_hashtag_does_not_fire():
    # #hiring alone (weight 2.4 < threshold 2.5) needs corroboration
    p = _post_with_body("Market update. #hiring")
    assert priority_ranker.has_recruiter_language(p) is False


def test_jobs_single_hashtag_does_not_fire():
    p = _post_with_body("Some industry news. #jobs")
    assert priority_ranker.has_recruiter_language(p) is False


def test_two_mid_weight_hashtags_fire():
    # #hiring (2.4) + #openposition (1.5) = 3.9 — well above 2.5
    p = _post_with_body("Latest opening. #hiring #openposition")
    assert priority_ranker.has_recruiter_language(p) is True


def test_hiring_plus_jobs_just_clears_threshold():
    # #hiring (2.4) + #jobs (0.6) = 3.0 → above threshold
    p = _post_with_body("Update. #hiring #jobs")
    assert priority_ranker.has_recruiter_language(p) is True


# ---------------------------- Public-API aliases stay back-compat ----------------------------

def test_is_hiring_intent_author_public_and_alias_identical():
    post = {"source": "linkedin_post", "author_title": "Talent Acquisition Lead"}
    assert (
        priority_ranker.is_hiring_intent_author(post)
        is priority_ranker._is_hiring_intent_author(post)
        is True
    )


def test_has_recruiter_language_public_and_alias_identical():
    post = {"source": "linkedin_post", "text": "We're hiring 3 PMs at Leap."}
    assert (
        priority_ranker.has_recruiter_language(post)
        is priority_ranker._has_recruiter_language(post)
        is True
    )


# ---------------------------- Connection-degree boost ----------------------------
# LinkedIn Connections CSV gate. Feature is opt-in: if the file doesn't exist,
# no connection reason ever fires. Tests write a scoped CSV at the configured
# path, then `reload_connections()` to pick up the new content.

def _write_connections_csv(rows: list[dict[str, str]]) -> None:
    """Write a minimal LinkedIn-export-shaped CSV (with the Notes: preamble
    LinkedIn actually prepends) to the configured path, then force a reload
    so the module picks up the new content."""
    import csv as _csv

    config.LINKEDIN_CONNECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.LINKEDIN_CONNECTIONS_PATH.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        # Real LinkedIn export has 3 preamble lines before the header.
        f.write("Notes:\n")
        f.write('"When exporting your connection data …"\n')
        f.write("\n")
        writer = _csv.DictWriter(
            f,
            fieldnames=[
                "First Name", "Last Name", "URL", "Email Address",
                "Company", "Position", "Connected On",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "First Name": r.get("first", ""),
                    "Last Name": r.get("last", ""),
                    "URL": r.get("url", ""),
                    "Email Address": r.get("email", ""),
                    "Company": r.get("company", ""),
                    "Position": r.get("position", ""),
                    "Connected On": r.get("connected_on", ""),
                }
            )
    priority_ranker.reload_connections()


@pytest.fixture
def connections_csv(tmp_path, monkeypatch):
    """Point the connections CSV at a temp path for each test so they're isolated."""
    monkeypatch.setattr(
        config, "LINKEDIN_CONNECTIONS_PATH", tmp_path / "connections.csv"
    )
    priority_ranker.reload_connections()
    yield
    priority_ranker.reload_connections()


def test_first_connection_fires_on_author_name_match(connections_csv):
    _write_connections_csv([{"first": "Priya", "last": "Nair", "company": "Razorpay"}])
    post = {"source": "linkedin_post", "author_name": "Priya Nair", "llm_score": 6.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "1st connection" in boosted[0]["priority_reasons"]
    assert "2nd connection" not in boosted[0]["priority_reasons"]  # no double-counting
    assert boosted[0]["priority_boost"] >= config.PRIORITY_BOOST_CONNECTION_1ST


def test_first_connection_case_insensitive(connections_csv):
    _write_connections_csv([{"first": "Priya", "last": "Nair", "company": "Razorpay"}])
    post = {"source": "linkedin_post", "author_name": "PRIYA NAIR", "llm_score": 6.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "1st connection" in boosted[0]["priority_reasons"]


def test_second_connection_via_shared_company(connections_csv):
    """Author works at a company where the user has a 1st-degree connection.
    Author himself is NOT a 1st-degree — fires 2nd only."""
    _write_connections_csv(
        [{"first": "Priya", "last": "Nair", "company": "Razorpay"}]
    )
    post = {
        "source": "linkedin_post",
        "author_name": "Vikram Patel",  # not in connections
        "author_title": "Head of Growth at Razorpay",
        "llm_score": 6.0,
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "2nd connection" in boosted[0]["priority_reasons"]
    assert "1st connection" not in boosted[0]["priority_reasons"]
    assert boosted[0]["priority_boost"] >= config.PRIORITY_BOOST_CONNECTION_2ND


def test_first_connection_short_circuits_second(connections_csv):
    """When an author is BOTH a 1st-degree connection AND works at a shared
    company, we award the 1st-degree boost only — not both."""
    _write_connections_csv(
        [{"first": "Priya", "last": "Nair", "company": "Razorpay"}]
    )
    post = {
        "source": "linkedin_post",
        "author_name": "Priya Nair",
        "author_title": "Head of Growth at Razorpay",
        "llm_score": 6.0,
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    reasons = boosted[0]["priority_reasons"]
    assert "1st connection" in reasons
    assert "2nd connection" not in reasons


def test_connection_signals_noop_without_csv(connections_csv):
    """If no CSV is loaded, neither 1st nor 2nd fires — feature is opt-in."""
    # connections_csv fixture points at a tmp_path with NO file — so when
    # reload_connections runs, the sets are empty.
    post = {
        "source": "linkedin_post",
        "author_name": "Priya Nair",
        "author_title": "Head of Growth at Razorpay",
        "llm_score": 6.0,
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    reasons = boosted[0]["priority_reasons"]
    assert "1st connection" not in reasons
    assert "2nd connection" not in reasons


def test_connection_signals_skip_non_linkedin_post_source(connections_csv):
    """Naukri / LinkedIn Jobs don't carry a real author_name — skip the check
    so we don't accidentally match a company name onto a generic listing."""
    _write_connections_csv([{"first": "Priya", "last": "Nair", "company": "Razorpay"}])
    post = {
        "source": "linkedin_job",
        "author_name": "Priya Nair",
        "author_title": "Head of Growth at Razorpay",
        "llm_score": 6.0,
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    reasons = boosted[0]["priority_reasons"]
    assert "1st connection" not in reasons
    assert "2nd connection" not in reasons


def test_company_legal_suffix_stripped_in_connections(connections_csv):
    """Export row 'Razorpay Software Pvt Ltd' should match author at 'Razorpay'."""
    _write_connections_csv(
        [{"first": "Priya", "last": "Nair", "company": "Razorpay Software Pvt Ltd"}]
    )
    post = {
        "source": "linkedin_post",
        "author_name": "Someone Else",
        "author_title": "Senior Designer at Razorpay",
        "llm_score": 6.0,
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "2nd connection" in boosted[0]["priority_reasons"]


def test_author_company_parses_at_pattern(connections_csv):
    """Direct test of the author-company extractor on the dominant headline
    shape: 'Role at Company'."""
    assert (
        priority_ranker._extract_author_company(
            {"author_title": "Head of Growth at Razorpay"}
        )
        == "razorpay"
    )


def test_author_company_ignores_at_scale_phrase(connections_csv):
    """'at scale' is a common phrase, not a company — the stopword list
    prevents a false positive."""
    result = priority_ranker._extract_author_company(
        {"author_title": "Engineering at scale"}
    )
    assert result != "scale"


def test_author_company_parses_pipe_separator(connections_csv):
    assert (
        priority_ranker._extract_author_company(
            {"author_title": "Head of Growth | Razorpay"}
        )
        == "razorpay"
    )


# ---------------------------- Off-archetype penalty ----------------------------
# The role_filter pipeline stage tags posts with role_filter_verdict == "pass"
# or "off_archetype" instead of dropping them. Priority ranker applies a
# penalty on the off_archetype tag so those posts still flow through every
# stage but need a strong boost stack to clear the final threshold.

def test_off_archetype_penalty_fires_when_tagged():
    """A post tagged off_archetype should carry a negative priority_boost
    equal in magnitude to config.PRIORITY_PENALTY_OFF_ARCHETYPE."""
    post = {
        "source": "naukri",
        "llm_score": 7.0,
        "role_filter_verdict": "off_archetype",
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    assert boosted[0]["priority_boost"] == pytest.approx(
        -config.PRIORITY_PENALTY_OFF_ARCHETYPE
    )
    assert "Off-archetype" in boosted[0]["priority_reasons"]


def test_off_archetype_penalty_skipped_when_pass():
    post = {
        "source": "naukri",
        "llm_score": 7.0,
        "role_filter_verdict": "pass",
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Off-archetype" not in boosted[0]["priority_reasons"]
    assert boosted[0]["priority_boost"] == 0.0


def test_off_archetype_penalty_skipped_when_tag_missing():
    """Back-compat: posts without the verdict tag (legacy runs, unit tests
    that don't go through the soft-filter stage) should behave as if the
    tag was 'pass' — no penalty."""
    post = {"source": "naukri", "llm_score": 7.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    assert "Off-archetype" not in boosted[0]["priority_reasons"]


def test_off_archetype_stackable_with_positive_boosts(tmp_path, monkeypatch):
    """A strong boost stack (Tier-1 + LinkedIn post) can outweigh the
    off-archetype penalty — exactly the user-intent trade-off: genuinely
    high-signal leads still reach the inbox even when off-archetype."""
    _write_tier1(["Razorpay"])
    post = {
        "source": "linkedin_post",
        "company": "Razorpay",
        "llm_score": 7.0,
        "role_filter_verdict": "off_archetype",
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    # +tier1 (1.0) + linkedin_post (0.6) = 1.6; penalty (-1.2) → net +0.4
    assert boosted[0]["priority_boost"] > 0
    # Both signals appear in the reasons list so the user sees the trade-off.
    reasons = boosted[0]["priority_reasons"]
    assert "Off-archetype" in reasons
    assert "Tier-1 Indian company" in reasons
    assert "LinkedIn post" in reasons


def test_off_archetype_penalty_clamps_at_zero_floor():
    """Final score is clamped to [0, 10] — a low-Haiku + off-archetype post
    should floor at 0, not go negative."""
    post = {
        "source": "naukri",
        "llm_score": 0.5,
        "role_filter_verdict": "off_archetype",
    }
    boosted = priority_ranker.apply_priority_boosts([post])
    assert boosted[0]["llm_score"] >= 0.0
    # Base + boost would be 0.5 - 1.2 = -0.7 without the clamp
    assert boosted[0]["base_llm_score"] == 0.5


# ---------------------------- priority_reasons_detailed ----------------------------
# The detailed shape carries (label, signed_magnitude) tuples so the email
# chip renderer can show `[India +0.4]` style numeric transparency.

def test_priority_reasons_detailed_populated():
    post = {"source": "linkedin_post", "llm_score": 7.0}
    boosted = priority_ranker.apply_priority_boosts([post])
    detailed = boosted[0]["priority_reasons_detailed"]
    assert isinstance(detailed, list)
    # Must be a list of (label, magnitude) tuples, not strings.
    assert all(isinstance(item, tuple) and len(item) == 2 for item in detailed)
    # The LinkedIn-post signal should be labeled + carry the configured boost.
    labels = [label for label, _ in detailed]
    mags = {label: mag for label, mag in detailed}
    assert "LinkedIn post" in labels
    assert mags["LinkedIn post"] == pytest.approx(config.PRIORITY_BOOST_LINKEDIN_POST)


def test_priority_reasons_detailed_and_legacy_reasons_aligned():
    """The legacy priority_reasons (list[str]) must stay in lockstep with
    priority_reasons_detailed (list[tuple]) — same labels, same order —
    so consumers reading either field see a consistent view."""
    post = {"source": "linkedin_post", "llm_score": 7.0}
    boosted = priority_ranker.apply_priority_boosts([post])[0]
    detailed_labels = [label for label, _ in boosted["priority_reasons_detailed"]]
    assert detailed_labels == boosted["priority_reasons"]


def test_priority_reasons_detailed_preserves_penalty_sign():
    """Penalty magnitudes must be stored NEGATIVE in the detailed list
    (not as absolute values) so renderers show `-1.2` instead of `+1.2`."""
    post = {
        "source": "naukri",
        "llm_score": 7.0,
        "role_filter_verdict": "off_archetype",
    }
    boosted = priority_ranker.apply_priority_boosts([post])[0]
    detailed = {label: mag for label, mag in boosted["priority_reasons_detailed"]}
    assert detailed["Off-archetype"] == pytest.approx(
        -config.PRIORITY_PENALTY_OFF_ARCHETYPE
    )


def test_compute_priority_boost_detailed_direct():
    """Direct test of the detailed helper — returns (boost, list[tuple])."""
    post = {"source": "linkedin_post", "llm_score": 7.0}
    boost, detailed = priority_ranker.compute_priority_boost_detailed(post)
    assert boost == pytest.approx(config.PRIORITY_BOOST_LINKEDIN_POST)
    assert detailed == [("LinkedIn post", config.PRIORITY_BOOST_LINKEDIN_POST)]


def test_compute_priority_boost_backcompat_returns_labels():
    """Legacy compute_priority_boost must still return list[str] labels so
    external callers (and ~30 existing assertions) don't break."""
    post = {"source": "linkedin_post", "llm_score": 7.0}
    boost, reasons = priority_ranker.compute_priority_boost(post)
    assert boost == pytest.approx(config.PRIORITY_BOOST_LINKEDIN_POST)
    assert reasons == ["LinkedIn post"]
    assert all(isinstance(r, str) for r in reasons)
