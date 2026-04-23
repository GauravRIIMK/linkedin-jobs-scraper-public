"""JD extraction across body, comments, nested replies, fetched external docs."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from src import jd_extractor


def _fake_response(body: str, ctype: str = "text/html; charset=utf-8", status: int = 200):
    r = MagicMock()
    r.text = body
    r.headers = {"content-type": ctype}
    r.status_code = status
    return r


# ---------- body-only ----------

def test_body_only_extracts_jd():
    post = {
        "text": "We are hiring Growth Marketer. 5+ years, SaaS experience. Bangalore.",
        "comments": [],
    }
    out = jd_extractor.extract_jd(post)
    assert "Growth Marketer" in out["jd_text"]
    assert out["jd_length"] > 0
    assert out["jd_doc_urls"] == []


def test_body_with_inline_urls_stripped():
    post = {"text": "Apply https://example.com/careers to join!", "comments": []}
    out = jd_extractor.extract_jd(post)
    assert "https://example.com" not in out["jd_text"]
    assert "Apply" in out["jd_text"]
    assert "to join" in out["jd_text"]


# ---------- comments and replies ----------

def test_comments_merged_into_jd():
    post = {
        "text": "Hiring SDE",
        "comments": [
            {"text": "Looking for 3+ years experience in Python and distributed systems."}
        ],
    }
    out = jd_extractor.extract_jd(post)
    assert "distributed systems" in out["jd_text"]
    assert out["jd_comment_count"] == 1


def test_short_comments_filtered_out():
    post = {
        "text": "Hiring SDE",
        "comments": [
            {"text": "Applied!"},  # <50 chars
            {"text": "interested"},  # <50 chars
            {"text": "This role needs strong knowledge of Python, Kafka, Postgres, system design, and on-call rotation."},
        ],
    }
    out = jd_extractor.extract_jd(post)
    # Reactions should not be in the merged JD
    assert "Applied" not in out["jd_text"]
    assert "Kafka" in out["jd_text"]


def test_nested_replies_walked():
    post = {
        "text": "Hiring",
        "comments": [
            {
                "text": "Can I apply as a fresher?",
                "replies": [
                    {"text": "Yes — we expect strong fundamentals in DSA and system design skills."}
                ],
            }
        ],
    }
    out = jd_extractor.extract_jd(post)
    assert out["jd_comment_count"] == 2  # original + reply
    assert "DSA" in out["jd_text"]


# ---------- private Google Docs / Drive ----------

def test_private_google_doc_captured_as_click_through():
    post = {
        "text": "JD here: https://docs.google.com/document/d/ABC123/edit",
        "comments": [],
    }
    out = jd_extractor.extract_jd(post)
    assert any("docs.google.com/document" in u for u in out["jd_private_docs"])
    # We should NOT try to fetch it
    assert all("docs.google.com" not in u for u in out["jd_fetched_urls"])


def test_private_drive_file_captured():
    post = {
        "text": "Role details: https://drive.google.com/file/d/XYZ/view",
        "comments": [],
    }
    out = jd_extractor.extract_jd(post)
    assert any("drive.google.com" in u for u in out["jd_private_docs"])


# ---------- fetchable ATS / career pages ----------

def test_lever_jd_fetched():
    post = {
        "text": "Senior PM role: https://jobs.lever.co/acme/abc-123",
        "comments": [],
    }
    html = (
        "<html><main>Senior Product Manager - 8+ years experience - Bangalore (hybrid) - "
        "ML/AI background required. You will own end-to-end product strategy, roadmap, and "
        "experimentation for our LLM-powered product suite, working cross-functionally with "
        "engineering, design, and data science. Must have prior experience shipping AI features "
        "to production at scale.</main></html>"
    )
    with patch("src.jd_extractor.requests.get", return_value=_fake_response(html)):
        out = jd_extractor.extract_jd(post)
    assert any("jobs.lever.co/acme/abc-123" in u for u in out["jd_fetched_urls"])
    assert "Senior Product Manager" in out["jd_text"]


def test_greenhouse_jd_fetched():
    post = {
        "text": "Apply: https://boards.greenhouse.io/somecompany/jobs/456",
        "comments": [],
    }
    html = (
        "<html><body><article>Staff Engineer needed. Strong Go and distributed systems "
        "background required. You'll lead architecture on our real-time data platform, "
        "mentor engineers, and drive technical decisions across multiple teams. "
        "7+ years of backend experience and experience running services at scale.</article>"
        "</body></html>"
    )
    with patch("src.jd_extractor.requests.get", return_value=_fake_response(html)):
        out = jd_extractor.extract_jd(post)
    assert "Staff Engineer" in out["jd_text"]


def test_ashby_jd_fetched():
    post = {
        "text": "Role: https://jobs.ashbyhq.com/company/some-role-id",
        "comments": [],
    }
    html = (
        "<html><body>Growth Marketing Lead role. Build and scale acquisition, retention, and "
        "lifecycle programs across paid, SEO, and product-led channels. React, SQL, and "
        "analytics stack required. 5+ years of B2B SaaS growth experience, ideally at a "
        "PLG startup. Remote-friendly with quarterly offsites.</body></html>"
    )
    with patch("src.jd_extractor.requests.get", return_value=_fake_response(html)):
        out = jd_extractor.extract_jd(post)
    assert "Growth Marketing Lead" in out["jd_text"]


def test_linkedin_urls_not_fetched():
    post = {
        "text": "More: https://www.linkedin.com/jobs/view/12345",
        "comments": [],
    }
    with patch("src.jd_extractor.requests.get") as mock_get:
        out = jd_extractor.extract_jd(post)
    mock_get.assert_not_called()
    assert out["jd_fetched_urls"] == []


def test_fetch_404_handled():
    post = {
        "text": "Role: https://jobs.lever.co/acme/abc",
        "comments": [],
    }
    with patch("src.jd_extractor.requests.get", return_value=_fake_response("Not found", status=404)):
        out = jd_extractor.extract_jd(post)
    assert out["jd_fetched_urls"] == []


def test_fetch_network_error_handled():
    post = {
        "text": "Role: https://jobs.lever.co/acme/abc",
        "comments": [],
    }
    import requests
    with patch("src.jd_extractor.requests.get", side_effect=requests.Timeout):
        out = jd_extractor.extract_jd(post)
    assert out["jd_fetched_urls"] == []
    # Body content should still be in jd_text
    assert "Role" in out["jd_text"]


def test_fetch_cap_respected():
    post = {
        "text": (
            "https://jobs.lever.co/a/1 "
            "https://jobs.lever.co/a/2 "
            "https://jobs.lever.co/a/3 "
            "https://jobs.lever.co/a/4"
        ),
        "comments": [],
    }
    call_count = {"n": 0}
    def fake(*a, **kw):
        call_count["n"] += 1
        return _fake_response("<main>JD text here with enough content for threshold.</main>")
    with patch("src.jd_extractor.requests.get", side_effect=fake):
        jd_extractor.extract_jd(post)
    # Cap is 2 per post
    assert call_count["n"] <= 2


def test_fetchable_in_comments():
    post = {
        "text": "Hiring a PM",
        "comments": [
            {"text": "Full JD: https://boards.greenhouse.io/xyz/jobs/9999"},
        ],
    }
    html = (
        "<main>Product Manager. 5+ years of experience required. FinTech preferred. "
        "You'll own lending product strategy, drive experimentation, and partner with "
        "engineering, design, compliance, and data teams on a $100M+ portfolio. "
        "Strong analytical background and prior regulated-industry experience a plus.</main>"
    )
    with patch("src.jd_extractor.requests.get", return_value=_fake_response(html)):
        out = jd_extractor.extract_jd(post)
    assert "FinTech" in out["jd_text"]
    assert any("boards.greenhouse.io/xyz/jobs/9999" in u for u in out["jd_fetched_urls"])


def test_empty_post_handled():
    out = jd_extractor.extract_jd({})
    assert out["jd_text"] == ""
    assert out["jd_doc_urls"] == []
    assert out["jd_comment_count"] == 0


def test_html_too_short_not_used():
    post = {
        "text": "https://jobs.lever.co/acme/role",
        "comments": [],
    }
    with patch("src.jd_extractor.requests.get", return_value=_fake_response("<html>tiny</html>")):
        out = jd_extractor.extract_jd(post)
    assert out["jd_fetched_urls"] == []
