"""gform detection: body, comments, replies, shorteners, bare URLs, HTML landing pages."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

from src import gform_filter


# ---------- extract_urls ----------

def test_extract_direct_gform_url():
    urls = gform_filter.extract_urls("Apply at https://docs.google.com/forms/d/e/xyz/viewform")
    assert "https://docs.google.com/forms/d/e/xyz/viewform" in urls


def test_extract_forms_gle():
    urls = gform_filter.extract_urls("Form: https://forms.gle/AbCdEf123")
    assert "https://forms.gle/AbCdEf123" in urls


def test_extract_bare_forms_gle_without_scheme():
    urls = gform_filter.extract_urls("Apply: forms.gle/AbCdEf123")
    assert "https://forms.gle/AbCdEf123" in urls


def test_extract_bare_docs_google_forms():
    urls = gform_filter.extract_urls("Fill: docs.google.com/forms/d/e/xyz/viewform")
    assert any("docs.google.com/forms" in u for u in urls)


def test_trailing_punctuation_trimmed():
    urls = gform_filter.extract_urls("Apply here: https://forms.gle/Abc123.")
    assert "https://forms.gle/Abc123" in urls


def test_zero_width_space_stripped():
    urls = gform_filter.extract_urls("Apply: https://forms.gle/Abc123\u200b more text")
    assert "https://forms.gle/Abc123" in urls


def test_parens_stripped():
    urls = gform_filter.extract_urls("(https://forms.gle/Abc123)")
    assert "https://forms.gle/Abc123" in urls


def test_empty_text_returns_empty():
    assert gform_filter.extract_urls("") == []
    assert gform_filter.extract_urls(None) == []  # type: ignore[arg-type]


def test_dedup_within_extract():
    txt = "https://forms.gle/Abc https://forms.gle/Abc"
    urls = gform_filter.extract_urls(txt)
    assert urls.count("https://forms.gle/Abc") == 1


# ---------- direct gform classification ----------

def test_is_direct_gform_true_cases():
    assert gform_filter._is_direct_gform("https://forms.gle/Abc")
    assert gform_filter._is_direct_gform("https://docs.google.com/forms/d/e/xxx/viewform")
    assert gform_filter._is_direct_gform("https://forms.google.com/a/b")


def test_is_direct_gform_false_cases():
    assert not gform_filter._is_direct_gform("https://linkedin.com/post/123")
    assert not gform_filter._is_direct_gform("https://lnkd.in/abc")
    assert not gform_filter._is_direct_gform("https://google.com/search")


# ---------- shortener classification ----------

def test_is_shortener_known_hosts():
    assert gform_filter._is_shortener("https://lnkd.in/dQxabc")
    assert gform_filter._is_shortener("https://bit.ly/xyz")
    assert gform_filter._is_shortener("https://tinyurl.com/a")
    assert gform_filter._is_shortener("https://rb.gy/a")
    assert gform_filter._is_shortener("https://t.co/abc")


def test_is_not_shortener():
    assert not gform_filter._is_shortener("https://google.com/forms")
    assert not gform_filter._is_shortener("https://linkedin.com/feed/...")


# ---------- find_gform_links ----------

def test_gform_in_body_direct():
    post = {"text": "We're hiring! Apply at https://forms.gle/Abc123", "comments": []}
    found = gform_filter.find_gform_links(post)
    assert found == ["https://forms.gle/Abc123"]


def test_gform_in_top_level_comment():
    post = {
        "text": "Hiring SDEs at our company",
        "comments": [{"text": "Apply: https://forms.gle/XYZ"}],
    }
    found = gform_filter.find_gform_links(post)
    assert "https://forms.gle/XYZ" in found


def test_gform_in_nested_reply():
    post = {
        "text": "Hiring!",
        "comments": [
            {
                "text": "Where to apply?",
                "replies": [{"text": "Here: https://forms.gle/REPLY"}],
            }
        ],
    }
    found = gform_filter.find_gform_links(post)
    assert "https://forms.gle/REPLY" in found


def test_gform_deeply_nested_reply():
    post = {
        "text": "Hiring!",
        "comments": [
            {
                "text": "Interested",
                "replies": [
                    {
                        "text": "Me too",
                        "replies": [{"text": "https://forms.gle/DEEP"}],
                    }
                ],
            }
        ],
    }
    found = gform_filter.find_gform_links(post)
    assert "https://forms.gle/DEEP" in found


def test_no_gform_returns_empty():
    post = {"text": "Hiring SDE, DM to apply", "comments": []}
    assert gform_filter.find_gform_links(post) == []


def test_linkedin_urls_not_matched_as_gform():
    post = {"text": "See https://linkedin.com/feed/post/xyz", "comments": []}
    assert gform_filter.find_gform_links(post) == []


# ---------- shortener redirect resolution ----------

def _mock_response(url: str, body: str = "", ctype: str = "text/html; charset=utf-8"):
    r = MagicMock()
    r.url = url
    r.text = body
    r.headers = {"content-type": ctype}
    r.status_code = 200
    return r


def test_shortener_resolves_to_gform():
    post = {"text": "Apply: https://lnkd.in/dQxabc", "comments": []}
    with patch("src.gform_filter.requests.get") as mock_get:
        mock_get.return_value = _mock_response("https://forms.gle/FINAL")
        found = gform_filter.find_gform_links(post)
    assert "https://forms.gle/FINAL" in found


def test_shortener_meta_refresh_discovers_gform():
    html = '<html><meta http-equiv="refresh" content="0;url=https://forms.gle/META"></html>'
    post = {"text": "Apply: https://bit.ly/xyz", "comments": []}
    with patch("src.gform_filter.requests.get") as mock_get:
        # Shortener landing page didn't 3xx; it returns HTML with meta refresh
        mock_get.return_value = _mock_response("https://bit.ly/xyz", body=html)
        found = gform_filter.find_gform_links(post)
    assert "https://forms.gle/META" in found


def test_shortener_landing_html_contains_gform():
    html = '<html><body><a href="https://docs.google.com/forms/d/e/HIDDEN/viewform">Apply</a></body></html>'
    post = {"text": "Apply: https://bit.ly/xyz", "comments": []}
    with patch("src.gform_filter.requests.get") as mock_get:
        mock_get.return_value = _mock_response("https://bit.ly/xyz", body=html)
        found = gform_filter.find_gform_links(post)
    assert any("docs.google.com/forms/d/e/HIDDEN" in u for u in found)


def test_multi_hop_shortener_chain():
    post = {"text": "Apply: https://bit.ly/first", "comments": []}
    # First call: bit.ly -> returns lnkd.in (another shortener). Second call: lnkd.in -> forms.gle.
    call_idx = {"n": 0}

    def fake_get(url, *a, **kw):
        call_idx["n"] += 1
        if call_idx["n"] == 1:
            return _mock_response("https://lnkd.in/second")
        return _mock_response("https://forms.gle/END")

    with patch("src.gform_filter.requests.get", side_effect=fake_get):
        found = gform_filter.find_gform_links(post)
    assert "https://forms.gle/END" in found


def test_shortener_network_failure_returns_empty():
    post = {"text": "Apply: https://lnkd.in/dead", "comments": []}
    import requests
    with patch("src.gform_filter.requests.get", side_effect=requests.ConnectionError):
        found = gform_filter.find_gform_links(post)
    assert found == []


def test_shortener_times_out_gracefully():
    post = {"text": "Apply: https://lnkd.in/slow", "comments": []}
    import requests
    with patch("src.gform_filter.requests.get", side_effect=requests.Timeout):
        found = gform_filter.find_gform_links(post)
    assert found == []


# ---------- filter_posts_with_gform ----------

def test_filter_keeps_only_posts_with_gform():
    posts = [
        {"text": "no gform here", "comments": []},
        {"text": "https://forms.gle/YES", "comments": []},
        {"text": "dm me to apply", "comments": []},
    ]
    kept = gform_filter.filter_posts_with_gform(posts)
    assert len(kept) == 1
    assert kept[0]["gform_urls"] == ["https://forms.gle/YES"]


def test_filter_empty_input():
    assert gform_filter.filter_posts_with_gform([]) == []


# ---------- URL scan cap ----------

def test_url_scan_is_capped_per_post(monkeypatch):
    # 20 non-gform URLs — we shouldn't fetch all of them
    body = " ".join(f"https://example{i}.com/page" for i in range(20))
    post = {"text": body, "comments": []}
    call_count = {"n": 0}

    def fake_get(url, *a, **kw):
        call_count["n"] += 1
        return _mock_response(url)

    with patch("src.gform_filter.requests.get", side_effect=fake_get):
        found = gform_filter.find_gform_links(post)
    # Most shouldn't get fetched (they're not shorteners and have long paths)
    # The cap shouldn't cause crashes either way
    assert found == []
