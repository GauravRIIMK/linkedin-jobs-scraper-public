"""Detect Google Forms in post body, comments, and nested replies.

Robustness layers (in order of aggressiveness):
1. Direct regex hit (docs.google.com/forms, forms.gle, forms.google.com).
2. Bare URL without scheme (`forms.gle/abc`) — prepend https:// and re-check.
3. Shortener host (lnkd.in, bit.ly, etc.) — follow up to 5 redirects, check final URL.
4. HTML body scan — some shorteners serve an HTML landing page instead of 3xx; parse
   `<meta http-equiv="refresh">` and any `forms.gle|docs.google.com/forms` hrefs.
5. Nested comment walk — recursively scan `replies` arrays, not just top-level comments.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import requests

from . import config

_URL_RE = re.compile(r"https?://[^\s)>\]\"'\u200b]+", re.IGNORECASE)
# Bare (no-scheme) patterns — prepended with https:// then deduped against full-URL matches
_BARE_FORM_RE = re.compile(
    r"(?:^|[\s(,\[])(forms\.gle/[A-Za-z0-9_-]+|docs\.google\.com/forms/[^\s)>\]\"']+)",
    re.IGNORECASE,
)
_DIRECT_RES = [re.compile(p, re.IGNORECASE) for p in config.GFORM_DIRECT_REGEXES]

# Substrings that, if present in any HTML body, mean there's a gform hidden inside
_GFORM_IN_HTML_RES = [
    re.compile(r"https?://forms\.gle/[A-Za-z0-9_\-]+", re.IGNORECASE),
    re.compile(r"https?://docs\.google\.com/forms/[^\s\"'<>]+", re.IGNORECASE),
    re.compile(r"https?://forms\.google\.com/[^\s\"'<>]+", re.IGNORECASE),
]
_META_REFRESH_RE = re.compile(
    r'<meta[^>]+http-equiv=["\']?refresh["\']?[^>]+url=([^"\'>\s]+)',
    re.IGNORECASE,
)

_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; GformHunter/1.1)",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_HTTP_TIMEOUT = 6.0
_MAX_REDIRECT_HOPS = 5
_MAX_URLS_SCANNED_PER_POST = 15  # hard cap to bound per-post latency


def extract_urls(text: str) -> list[str]:
    if not text:
        return []
    # Full URLs
    urls = [_trim_trailing(u) for u in _URL_RE.findall(text)]
    # Bare URLs — prepend https://
    for match in _BARE_FORM_RE.findall(text):
        bare = _trim_trailing(match)
        if bare and not bare.startswith("http"):
            urls.append("https://" + bare)
    # Dedupe, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _trim_trailing(url: str) -> str:
    while url and url[-1] in ".,);]}\"'\u200b":
        url = url[:-1]
    return url


def _is_direct_gform(url: str) -> bool:
    return any(rx.search(url) for rx in _DIRECT_RES)


def _is_shortener(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False
    return any(host == s or host.endswith("." + s) for s in config.SHORTENER_HOSTS)


def resolve_redirect(url: str) -> tuple[str, str]:
    """Follow redirects and also return the final HTML body for scanning.

    Returns (final_url_after_redirects, body_text_or_empty).
    """
    try:
        resp = requests.get(
            url,
            allow_redirects=True,
            timeout=_HTTP_TIMEOUT,
            headers=_REQUEST_HEADERS,
        )
        body = resp.text if "html" in resp.headers.get("content-type", "").lower() else ""
        return (resp.url or url, body[:40_000])
    except requests.RequestException:
        return (url, "")


def _gform_in_html(html: str) -> str | None:
    """Scan HTML (and any meta-refresh target) for a direct Google Form URL."""
    if not html:
        return None
    for rx in _GFORM_IN_HTML_RES:
        m = rx.search(html)
        if m:
            return _trim_trailing(m.group(0))
    # Also check meta-refresh targets (some shorteners do this)
    m = _META_REFRESH_RE.search(html)
    if m:
        candidate = _trim_trailing(m.group(1))
        for rx in _GFORM_IN_HTML_RES:
            if rx.search(candidate):
                return candidate
    return None


def _walk_comment_texts(comments: list[Any]) -> list[str]:
    """Flatten comments + nested replies."""
    out: list[str] = []
    for c in comments or []:
        if not isinstance(c, dict):
            continue
        txt = c.get("text") or ""
        if txt:
            out.append(txt)
        replies = c.get("replies")
        if isinstance(replies, list):
            out.extend(_walk_comment_texts(replies))
    return out


def find_gform_links(post: dict[str, Any]) -> list[str]:
    """Return confirmed Google Form URLs found anywhere in the post.

    Looks in: body + top-level comments + nested replies. Resolves shorteners and
    scans HTML bodies for hidden gform links. Returns direct form URLs (not shortened).
    """
    text_sources: list[str] = [post.get("text") or ""]
    text_sources.extend(_walk_comment_texts(post.get("comments") or []))

    # Collect candidate URLs across all sources, dedupe, cap total count
    candidates: list[str] = []
    seen_urls: set[str] = set()
    for source in text_sources:
        for url in extract_urls(source):
            if url not in seen_urls:
                seen_urls.add(url)
                candidates.append(url)
                if len(candidates) >= _MAX_URLS_SCANNED_PER_POST:
                    break
        if len(candidates) >= _MAX_URLS_SCANNED_PER_POST:
            break

    found: list[str] = []
    seen: set[str] = set()
    for url in candidates:
        resolved = _resolve_to_gform(url)
        if resolved and resolved not in seen:
            seen.add(resolved)
            found.append(resolved)

    return found


def _resolve_to_gform(url: str, hops: int = 0) -> str | None:
    """Recursively try to reach a direct gform URL. Returns None if we can't."""
    if hops > _MAX_REDIRECT_HOPS:
        return None
    if _is_direct_gform(url):
        return url
    # Only fetch when the URL is a known shortener, OR a short-pathed URL
    # (heuristic for custom-domain shorteners like company.tld/apply).
    should_fetch = _is_shortener(url)
    if not should_fetch:
        try:
            parsed = urlparse(url)
            path = parsed.path.strip("/")
            should_fetch = 0 < len(path) <= 40 and "." not in path
        except ValueError:
            return None
    if not should_fetch:
        return None

    final_url, html = resolve_redirect(url)
    if _is_direct_gform(final_url):
        return final_url
    hit = _gform_in_html(html)
    if hit:
        return hit
    # One more hop if the final URL still looks like a shortener and differs
    if final_url and final_url != url and _is_shortener(final_url):
        return _resolve_to_gform(final_url, hops=hops + 1)
    return None


def filter_posts_with_gform(
    posts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only posts that contain at least one Google Form URL (direct or via shortener)."""
    kept: list[dict[str, Any]] = []
    for p in posts:
        gforms = find_gform_links(p)
        if gforms:
            p = {**p, "gform_urls": gforms}
            kept.append(p)
    return kept
