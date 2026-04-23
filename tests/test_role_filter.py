"""Role-archetype gate — blocks SWE / ML engineer / data scientist JDs from
the ai_ops_automation track (user is vibecoder, not a coder)."""
from __future__ import annotations

import pytest

from src import role_filter


def _post(title: str, body: str = "") -> dict:
    return {"author_title": title, "jd_text": body}


# ---------------------------- ai_ops_automation ----------------------------

@pytest.mark.parametrize(
    "title",
    [
        "Senior Software Engineer",
        "Software Developer",
        "SDE-2",
        "SDE 3",
        "SDE III",
        "Backend Engineer at Razorpay",
        "Full Stack Developer",
        "Full-Stack Engineer",
        "Frontend Developer",
        "Android Developer",
        "iOS Developer",
        "ML Engineer",
        "Machine Learning Engineer",
        "Data Scientist - Pricing",
        "Data Engineer",
        "Research Engineer",
        "Research Scientist",
        "Applied Scientist",
        "DevOps Engineer",
        "Site Reliability Engineer",
        "SRE",
        "Platform Engineer",
        "Python Developer",
        "React Developer",
        "Golang Developer",
        "Go Developer",
        "Principal Engineer",
        "Staff Engineer",
        "SDET",
        "QA Engineer",
        "NLP Engineer",
        "Computer Vision Engineer",
        "MLOps Engineer",
        "Node.js Developer",
        "Nodejs Developer",
        "PHP Developer",
    ],
)
def test_rejects_coder_titles_for_ai_ops_automation(title):
    assert not role_filter.passes_role_filter("ai_ops_automation", _post(title))


@pytest.mark.parametrize(
    "title",
    [
        "AI Product Manager",
        "AI Program Manager",
        "Chief of Staff",
        "AI Operator",
        "AI Transformation Lead",
        "Automation Specialist",
        "Head of Automation",
        "Director of Automation",
        "AI Consultant",
        "Solutions Architect",
        "Growth Marketer",
        "Content Writer",
        "Business Analyst",
        "Operations Manager",
        "Product Marketing Manager",
        "Category Manager",
        "Associate Principal",  # contains "principal" but not "principal engineer"
    ],
)
def test_keeps_non_coder_titles_for_ai_ops_automation(title):
    assert role_filter.passes_role_filter("ai_ops_automation", _post(title))


def test_positive_rescues_borderline_title():
    """A title that hits a negative but also matches a positive should pass."""
    # "AI Product Manager" doesn't hit any negatives — baseline pass.
    # "Automation Specialist - Python Developer role" DOES hit 'python developer'
    # in negatives, but 'automation specialist' in positives rescues it.
    post = _post("Automation Specialist - Python Developer scripting needed")
    assert role_filter.passes_role_filter("ai_ops_automation", post)


def test_title_window_includes_body_prefix():
    """When author_title is empty, we fall back to the first 240 chars of jd_text."""
    post = _post("", "Software Engineer role. We're looking for great engineers.")
    assert not role_filter.passes_role_filter("ai_ops_automation", post)


def test_jd_framing_prefix_stripped():
    """extract_jd prepends markers like '[post body]' and '[from comments]'.
    Those shouldn't prevent our regex from matching — we strip them before scan."""
    post = _post(
        "",
        "[post body]\nSenior ML Engineer opening — build recommender systems.",
    )
    assert not role_filter.passes_role_filter("ai_ops_automation", post)


def test_deep_body_mention_does_not_trigger():
    """A passing role title + a deep-body 'software engineer' mention should
    not reject — we only scan the title window (first ~240 chars of body)."""
    long_body = (
        "AI Product Manager opening at a fintech. You'll own the AI roadmap, "
        "work with product marketers, design team, and data scientists. "
        + ("filler " * 60)  # push further content well past 240 chars
        + "Collaborate closely with our software engineer team on integration."
    )
    post = _post("AI Product Manager", long_body)
    assert role_filter.passes_role_filter("ai_ops_automation", post)


# ---------------------------- Other tracks ----------------------------

def test_tracks_without_negatives_pass_everything():
    """strategy_consulting / product_brand_category don't ship with negatives —
    every title passes for those tracks. (growth_marketing now has digital /
    performance marketing negatives — covered separately below.)"""
    assert role_filter.passes_role_filter("strategy_consulting", _post("ML Engineer"))
    assert role_filter.passes_role_filter("product_brand_category", _post("Data Scientist"))


def test_growth_marketing_passes_swe_titles():
    """Non-marketing titles have no rule hit in the growth_marketing track —
    the filter is scoped to the digital/performance marketing archetype only."""
    assert role_filter.passes_role_filter("growth_marketing", _post("Software Engineer"))
    assert role_filter.passes_role_filter("growth_marketing", _post("Data Scientist"))


def test_unknown_track_passes():
    assert role_filter.passes_role_filter("unknown_track", _post("SDE"))


# ---------------------------- growth_marketing negatives ----------------------------
# User explicitly asked: "strictly avoid" digital + performance marketing roles
# because they have zero experience in those sub-specializations.

@pytest.mark.parametrize(
    "title",
    [
        "Digital Marketing Manager",
        "Digital Marketing Executive",
        "Digital Marketer",
        "Performance Marketing Manager",
        "Performance Marketer",
        "Performance Marketing Lead",
        "Paid Media Specialist",
        "Paid Ads Manager",
        "Paid Acquisition Lead",
        "Paid Social Manager",
        "Media Buyer",
        "Media Planner",
        "PPC Specialist",
        "PPC Manager",
        "Google Ads Specialist",
        "Google Ads Manager",
        "Meta Ads Specialist",
        "Facebook Ads Manager",
        "SEM Specialist",
        "SEM Manager",
        "SEO Specialist",
        "SEO Manager",
        "SEO Executive",
        "SEO Analyst",
        "Programmatic Manager",
        "Ad Operations Lead",
        "Ad Ops Manager",
        "Adwords Specialist",
    ],
)
def test_rejects_digital_performance_marketing_titles(title):
    assert not role_filter.passes_role_filter("growth_marketing", _post(title))


@pytest.mark.parametrize(
    "title",
    [
        "Growth Marketing Manager",
        "Head of Growth",
        "Growth Hacker",
        "Brand Marketing Manager",
        "Brand Manager",
        "Product Marketing Manager",
        "PMM",
        "Content Marketing Lead",
        "Content Strategist",
        "Lifecycle Marketing Manager",
        "Retention Marketing Manager",
        "CRM Marketing Lead",
        "Category Marketing Manager",
        "Category Manager",
        "Integrated Marketing Manager",
        "Marketing Director",
        "Head of Marketing",
        "CMO",
        "VP Marketing",
        "Consumer Insights Lead",
    ],
)
def test_keeps_broader_marketing_titles(title):
    assert role_filter.passes_role_filter("growth_marketing", _post(title))


def test_growth_marketing_positive_rescues_mixed_role():
    """A Growth Marketing Manager JD that MENTIONS digital/performance marketing
    as a nice-to-have should NOT be rejected — the positive rescues it."""
    post = _post(
        "Growth Marketing Manager",
        "We're hiring a growth marketing manager; digital marketing exposure a plus.",
    )
    assert role_filter.passes_role_filter("growth_marketing", post)


def test_growth_marketing_rejects_pure_performance_marketing_even_if_body_mentions_growth():
    """Body-level 'growth' mentions alone don't save a Performance Marketing
    title — the title window decision dominates."""
    post = _post(
        "Performance Marketing Manager",
        "Own paid media strategy. Growth thinking a bonus.",
    )
    assert not role_filter.passes_role_filter("growth_marketing", post)


# ---------------------------- Body-declarative negatives ----------------------------
# The recruiter posted "Hey network! At Acme we're building the future ..."
# and the role title "Digital Marketing Manager" sits at char 600+. The
# title-window check misses it; the body-declarative check should catch it.

@pytest.mark.parametrize(
    "body",
    [
        # Standard recruiter opening + role statement deep in the body.
        "Hello network! " + ("filler " * 60) + "We're hiring a Digital Marketing Manager to own paid media.",
        "Excited to share that " + ("filler " * 60) + "we are looking for a Performance Marketing Lead.",
        "At Acme we're building the future. " + ("filler " * 60) + "Role: Digital Marketing Manager.",
        "Big news — " + ("filler " * 60) + "Position: SEO Specialist with 3+ years experience.",
        "We just raised Series B. " + ("filler " * 60) + "Job Title: Paid Media Manager.",
        "Hey everyone — " + ("filler " * 60) + "we need a Google Ads Manager to scale accounts.",
    ],
)
def test_body_declarative_rejects_digital_performance_marketing(body):
    """Even when the author_title is empty and the body has long preamble,
    a declarative 'hiring a {digital|performance|SEO|paid} …' deep in the
    body should reject the post for growth_marketing track."""
    assert not role_filter.passes_role_filter("growth_marketing", _post("", body))


def test_body_casual_mention_still_passes():
    """A casual 'digital marketing exposure a plus' mention in body must
    NOT reject — only role-declarative preambles fire the body check."""
    post = _post(
        "Growth Marketing Manager",
        "We're a Series C startup. "
        + ("filler " * 100)
        + "Digital marketing exposure is a nice-to-have, not required.",
    )
    assert role_filter.passes_role_filter("growth_marketing", post)


def test_body_declarative_for_ai_ops_automation_coder_role():
    """ai_ops_automation rejects 'hiring a Software Engineer' even with a
    long recruiter preamble and no title field."""
    body = "Welcome to my post! " + ("filler " * 60) + "We are hiring a Senior Software Engineer to build platform infra."
    assert not role_filter.passes_role_filter("ai_ops_automation", _post("", body))


# ---------------------------- filter_by_role ----------------------------

def test_filter_by_role_counts_and_order():
    posts = [
        _post("AI Product Manager"),
        _post("Software Engineer"),
        _post("Chief of Staff"),
        _post("ML Engineer"),
    ]
    kept, rejected = role_filter.filter_by_role("ai_ops_automation", posts)
    kept_titles = [p["author_title"] for p in kept]
    assert kept_titles == ["AI Product Manager", "Chief of Staff"]
    assert rejected == 2


# ---------------------------- Word boundaries ----------------------------

def test_sde_not_matched_inside_designer():
    """'designer' contains 'des' not 'sde' as substring, but word boundaries
    matter for phrases like 'sde' where the letters might incidentally appear."""
    post = _post("UX Designer")
    assert role_filter.passes_role_filter("ai_ops_automation", post)


def test_matches_across_hyphenation():
    """'full-stack developer' and 'full stack developer' are the same role."""
    assert not role_filter.passes_role_filter("ai_ops_automation", _post("Full-Stack Developer"))
    assert not role_filter.passes_role_filter("ai_ops_automation", _post("Full Stack Developer"))
    assert not role_filter.passes_role_filter("ai_ops_automation", _post("Fullstack-Developer"))
