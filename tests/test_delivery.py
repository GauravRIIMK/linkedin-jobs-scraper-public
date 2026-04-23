"""Delivery: CSV writing, subject line, HTML email, Resend send + retry."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src import config, delivery


def _sample_matches() -> dict[str, list[dict]]:
    return {
        "growth_marketing": [
            {
                "post_url": "https://linkedin.com/feed/update/urn:li:activity:1/",
                "post_urn": "urn:li:activity:1",
                "author_name": "Acme HR",
                "author_title": "Recruiter at Acme",
                "posted_at": "2026-04-20T10:00:00Z",
                "llm_score": 8.5,
                "similarity": 0.75,
                "llm_reason": "Strong growth marketing fit with SaaS experience.",
                "gform_urls": ["https://forms.gle/ABC"],
                "jd_doc_urls": [],
                "jd_text": "Growth Marketing Manager needed for Bangalore office.",
            }
        ],
        "product_ai": [],
    }


def test_write_csv_creates_file_with_headers(tmp_path):
    matches = _sample_matches()
    out = tmp_path / "out.csv"
    delivery.write_csv(matches, out)
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["track"] == "growth_marketing"
    assert rows[0]["post_url"].startswith("https://linkedin.com")
    assert rows[0]["llm_score"] == "8.5"


def test_write_csv_handles_empty_tracks(tmp_path):
    out = tmp_path / "empty.csv"
    delivery.write_csv({"growth": []}, out)
    content = out.read_text(encoding="utf-8")
    # Header present but no data rows
    assert "track," in content
    assert content.count("\n") == 1


def test_build_subject_with_matches():
    s = delivery.build_subject(_sample_matches())
    assert "1 Growth Marketing" in s
    # Subject now uses today's ISO date (not "week of …") because the cron is daily.
    assert "week of" not in s.lower()
    from datetime import date
    assert date.today().isoformat() in s


def test_build_subject_no_matches():
    s = delivery.build_subject({"growth": [], "product": []})
    assert "no matches" in s.lower()
    assert "week of" not in s.lower()


def test_build_html_is_daily_not_weekly():
    """The body header used to say 'Weekly LinkedIn matches' — must now say daily.
    Also: the 'next week' language in the footer must be gone."""
    html = delivery.build_html(_sample_matches())
    low = html.lower()
    assert "weekly" not in low
    assert "next week" not in low
    assert "last 7 days" not in low  # old hardcoded wording
    assert "daily linkedin matches" in low


def test_build_html_empty_tracks_use_daily_wording():
    html = delivery.build_html({"growth_marketing": [], "product_ai": []})
    assert "No matches today" in html
    assert "No matches this week" not in html


def test_build_html_renders_priority_badges():
    matches = {
        "growth_marketing": [
            {
                "post_url": "https://linkedin.com/x",
                "author_name": "Acme",
                "llm_score": 8.0,
                "base_llm_score": 7.0,
                "priority_boost": 1.0,
                "priority_reasons": ["Tier-1 Indian company", "Remote"],
                "llm_reason": "strong fit",
                "jd_text": "Hiring growth marketer",
            }
        ]
    }
    html = delivery.build_html(matches)
    assert "Tier-1 Indian company" in html
    assert "Remote" in html
    # Boost should appear next to the score
    assert "+1.0" in html


def test_build_html_contains_post_link_and_form_link():
    html = delivery.build_html(_sample_matches())
    assert "https://linkedin.com/feed/update/urn:li:activity:1/" in html
    assert "https://forms.gle/ABC" in html
    assert "Acme HR" in html
    assert "8.5" in html


def test_build_html_escapes_untrusted_input():
    malicious = {
        "growth": [
            {
                "post_url": "https://linkedin.com/x",
                "author_name": "<script>alert(1)</script>",
                "llm_score": 7.0,
                "similarity": 0.7,
                "llm_reason": "<img src=x onerror=alert(2)>",
                "jd_text": "<b>bold</b>",
            }
        ]
    }
    html = delivery.build_html(malicious)
    # Raw script tags must not survive
    assert "<script>" not in html
    assert "</script>" not in html
    # Image/event-handler injection must be escaped
    assert "<img src=x onerror=" not in html
    # The payload must appear in its HTML-escaped form
    assert "&lt;script&gt;" in html
    assert "&lt;img" in html
    # Trusted body tags built by build_html itself are fine to see
    assert "<html" in html.lower()


# ---------- Resend backend ----------

def test_send_via_resend_success():
    resp = MagicMock()
    resp.status_code = 200
    resp.text = '{"id":"ok"}'
    with patch("src.delivery.requests.post", return_value=resp) as mock_post:
        delivery._send_via_resend("subj", "<p>hi</p>", "to@x.com", None)
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert "api.resend.com" in args[0]
    assert kwargs["headers"]["Authorization"].startswith("Bearer ")
    assert kwargs["json"]["to"] == ["to@x.com"]
    assert kwargs["json"]["subject"] == "subj"


def test_send_via_resend_retries_on_500():
    failing = MagicMock(); failing.status_code = 502; failing.text = "bad gateway"
    success = MagicMock(); success.status_code = 200; success.text = "{}"
    with patch("src.delivery.requests.post", side_effect=[failing, success]):
        with patch("src.delivery.time.sleep"):  # skip real backoff
            delivery._send_via_resend("s", "h", "to@x.com", None)


def test_send_via_resend_fails_fast_on_401():
    bad = MagicMock(); bad.status_code = 401; bad.text = "invalid api key"
    with patch("src.delivery.requests.post", return_value=bad):
        with pytest.raises(RuntimeError, match="401"):
            delivery._send_via_resend("s", "h", "to@x.com", None)


def test_send_via_resend_gives_up_after_3_attempts():
    bad = MagicMock(); bad.status_code = 503; bad.text = "busy"
    with patch("src.delivery.requests.post", return_value=bad):
        with patch("src.delivery.time.sleep"):
            with pytest.raises(RuntimeError, match="3 attempts"):
                delivery._send_via_resend("s", "h", "to@x.com", None)


def test_send_via_resend_with_attachment(tmp_path):
    f = tmp_path / "out.csv"
    f.write_text("track,url\ngrowth,x\n")
    resp = MagicMock(); resp.status_code = 200; resp.text = "{}"
    with patch("src.delivery.requests.post", return_value=resp) as mock_post:
        delivery._send_via_resend("s", "h", "to@x.com", f)
    kwargs = mock_post.call_args.kwargs
    assert kwargs["json"]["attachments"][0]["filename"] == "out.csv"


def test_resend_requires_api_key(monkeypatch):
    monkeypatch.setattr(config, "RESEND_API_KEY", None)
    with pytest.raises(RuntimeError, match="RESEND_API_KEY"):
        delivery._send_via_resend("s", "h", "to@x.com", None)


# ---------- deliver() orchestration ----------

def test_deliver_writes_csv_and_sends(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    matches = _sample_matches()
    resp = MagicMock(); resp.status_code = 200; resp.text = "{}"
    with patch("src.delivery.requests.post", return_value=resp):
        info = delivery.deliver(matches)
    assert info["sent"] is True
    assert Path(info["csv_path"]).exists()
    assert info["method"] == "resend"


def test_deliver_writes_error_file_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    matches = _sample_matches()
    bad = MagicMock(); bad.status_code = 401; bad.text = "bad key"
    with patch("src.delivery.requests.post", return_value=bad):
        info = delivery.deliver(matches)
    assert info["sent"] is False
    err_files = list(tmp_path.glob("delivery_error_*.txt"))
    assert err_files
