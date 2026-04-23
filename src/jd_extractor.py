"""Extract a clean JD string from a post — 3 sources: body, linked docs, comments.

Strategy:
1. **Body text** — primary JD source (stripped of URLs).
2. **Comments** — recruiters often drop the JD or the application link in a pinned comment.
   We also walk nested `replies` if the actor surfaces them.
3. **Linked docs** — Google Docs published to web, Notion public pages, Lever/Greenhouse/Ashby
   ATS listings, company career pages. We attempt a short HTTP fetch and extract readable text.
   Private Google Docs/Drive (auth-walled) are detected and surfaced as clickable links
   without fetching.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import requests

# Google Docs/Drive URLs are kept as "click-through" if we can't fetch content
_PRIVATE_DOC_RE = re.compile(
    r"https?://(?:docs|drive)\.google\.com/(?:document|file|presentation|spreadsheets)/[^\s)>\]\"']+",
    re.IGNORECASE,
)

# JD-bearing destinations we *do* try to fetch (ATS pages, public Notion, company careers)
_FETCHABLE_JD_RE = re.compile(
    r"https?://(?:"
    r"jobs\.lever\.co|"
    r"boards\.greenhouse\.io|"
    r"jobs\.ashbyhq\.com|"
    r"apply\.workable\.com|"
    r"[a-z0-9\-]+\.notion\.site|"
    r"notion\.so|"
    r"www\.notion\.so|"
    r"[a-z0-9\-]+\.teamtailor\.com|"
    r"[a-z0-9\-]+\.recruitee\.com|"
    r"wellfound\.com/company|"
    r"careers\.[a-z0-9\-]+\.[a-z]{2,}|"
    r"[a-z0-9\-]+\.breezy\.hr|"
    r"app\.dover\.io"
    r")[^\s)>\]\"']+",
    re.IGNORECASE,
)

_URL_RE = re.compile(r"https?://[^\s)>\]\"'\u200b]+")

# LinkedIn/noise links we never fetch as JDs
_BLOCKED_HOSTS = (
    "linkedin.com",
    "lnkd.in",
    "youtube.com",
    "youtu.be",
    "twitter.com",
    "x.com",
    "instagram.com",
    "facebook.com",
)

# Reasonable caps (tuned for daily cron within 30-min GH Actions timeout)
_MAX_FETCH_URLS_PER_POST = 2
_FETCH_TIMEOUT_S = 6
_FETCH_MAX_CHARS = 2500
_MIN_COMMENT_LEN_FOR_JD = 50  # skip "applied!" / "interested!" reactions


def extract_jd(post: dict[str, Any]) -> dict[str, Any]:
    body = (post.get("text") or "").strip()
    comments = post.get("comments") or []

    # Gather all text: body + comments + (nested) replies
    all_comment_texts = _walk_comment_texts(comments)
    comment_blob = "\n\n".join(all_comment_texts)

    # Collect JD-bearing URLs across all text
    full_text = f"{body}\n\n{comment_blob}"
    private_docs = _dedupe(_PRIVATE_DOC_RE.findall(full_text))
    fetchable = _dedupe(_FETCHABLE_JD_RE.findall(full_text))

    # Fetch external JD content (best-effort)
    fetched_sections: list[str] = []
    fetched_urls: list[str] = []
    for url in fetchable[:_MAX_FETCH_URLS_PER_POST]:
        if _host_blocked(url):
            continue
        content = _try_fetch_readable(url)
        if content:
            fetched_sections.append(f"[from {url}]\n{content}")
            fetched_urls.append(url)

    # Compose merged JD text for embedding/rerank
    parts: list[str] = []
    if body:
        parts.append(f"[post body]\n{_strip_urls(body)}")
    jd_rich_comments = [c for c in all_comment_texts if len(c) >= _MIN_COMMENT_LEN_FOR_JD]
    if jd_rich_comments:
        joined = "\n".join(_strip_urls(c) for c in jd_rich_comments[:5])
        parts.append(f"[from comments]\n{joined}")
    parts.extend(fetched_sections)

    merged = "\n\n---\n\n".join(parts).strip()

    # Backward-compat field: jd_doc_urls = private docs + successfully-fetched pages
    jd_doc_urls = _dedupe(private_docs + fetched_urls)

    return {
        "jd_text": merged,
        "jd_length": len(merged),
        "jd_doc_urls": jd_doc_urls,
        "jd_private_docs": private_docs,
        "jd_fetched_urls": fetched_urls,
        "jd_comment_count": len(all_comment_texts),
    }


# ---------- helpers ----------


def _walk_comment_texts(comments: list[Any]) -> list[str]:
    """Flatten comments + nested replies (some actors return `replies` arrays)."""
    out: list[str] = []
    for c in comments or []:
        if not isinstance(c, dict):
            continue
        txt = (c.get("text") or "").strip()
        if txt:
            out.append(txt)
        replies = c.get("replies")
        if isinstance(replies, list):
            out.extend(_walk_comment_texts(replies))
    return out


def _strip_urls(text: str) -> str:
    return _URL_RE.sub("", text).strip()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for i in items:
        key = i.rstrip(".,);]}\"'\u200b")
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _host_blocked(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return True
    return any(host == b or host.endswith("." + b) for b in _BLOCKED_HOSTS)


def _try_fetch_readable(url: str) -> str | None:
    """Fetch a URL with a short timeout and return extracted text (best-effort)."""
    try:
        resp = requests.get(
            url,
            timeout=_FETCH_TIMEOUT_S,
            allow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; JDHunter/1.0)",
                "Accept": "text/html,application/xhtml+xml,text/plain,*/*",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
    except requests.RequestException:
        return None

    ctype = resp.headers.get("content-type", "").lower()
    if resp.status_code >= 400:
        return None
    if "text" not in ctype and "html" not in ctype and "json" not in ctype:
        return None

    text = _extract_text_from_html(resp.text) if "html" in ctype else resp.text
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 120:
        return None
    return text[:_FETCH_MAX_CHARS]


def _extract_text_from_html(html: str) -> str:
    """Lightweight HTML → text. Uses BeautifulSoup if available, regex fallback otherwise."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "svg"]):
            tag.decompose()
        # Prefer <main>/<article> content when present
        main = soup.find("main") or soup.find("article") or soup.body or soup
        return main.get_text(separator=" ", strip=True)
    except Exception:
        # Strip tags crudely
        no_tags = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
        no_tags = re.sub(r"<style[\s\S]*?</style>", " ", no_tags, flags=re.I)
        no_tags = re.sub(r"<[^>]+>", " ", no_tags)
        return re.sub(r"&[a-z#0-9]+;", " ", no_tags, flags=re.I)
