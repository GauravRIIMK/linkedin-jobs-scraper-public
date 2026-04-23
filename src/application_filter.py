"""Detect machine-actionable application methods on hiring posts.

Supersedes the gform-only gate with four signal types:

  1. Google Form       — delegates to `gform_filter.find_gform_links()`
                          (preserves shortener resolution + nested comment walk)
  2. ATS platforms     — 19 platforms via hostname-anchored regex
                          (Lever, Greenhouse, Ashby, Workday, Workable, SmartRecruiters,
                           Recruitee, Teamtailor, Jobvite, iCIMS, BambooHR, Keka,
                           Zoho Recruit, Wellfound/AngelList, Breezy, Dover,
                           Naukri, Instahyre, Cutshort)
  3. LinkedIn Easy Apply — a `linkedin.com/jobs/view/<id>` URL, OR an explicit
                            "easy apply" / "apply on linkedin" phrase.
  4. Mail-to-apply     — requires BOTH a non-noreply email AND a mail-intent
                          phrase ("mail us at …", "send your CV", "drop a mail", …)
                          to avoid false positives on random footer signatures.

A post is kept if ANY signal fires. Each kept post is annotated with:
    apply_methods : list[str]  (e.g. ["gform", "lever"])
    apply_urls    : list[str]  (all URLs across all detected methods)
    apply_emails  : list[str]  (only when mail-to-apply fires)
    gform_urls    : list[str]  (preserved for back-compat with delivery/CSV)

Design notes on second-order effects:
  - We deliberately keep `gform_urls` populated even when there's no gform,
    so downstream CSV/HTML writers that read `m["gform_urls"] or []` still work.
  - We never require LOCATION in this filter — locality fit is delegated to
    the Haiku reranker, which sees the full resume summary and the JD.
  - Blocklist strips obvious automated senders (noreply, support, info, ...)
    before requiring a co-located mail-intent phrase; this avoids matching
    the recruiter's own noreply@ signature.
"""
from __future__ import annotations

import re
from typing import Any

from . import config, gform_filter

# Imported lazily inside functions to avoid an eager cycle during module
# reload in tests; priority_ranker doesn't import us so there's no real
# cycle, but keeping the import local makes the dependency direction
# explicit (soft-apply ONLY exists because of these signals).
_PRIORITY_RANKER = None


def _priority_ranker():
    """Lazy import so test harnesses that monkeypatch priority_ranker
    pick up the patched version instead of a cached one."""
    global _PRIORITY_RANKER
    if _PRIORITY_RANKER is None:
        from . import priority_ranker
        _PRIORITY_RANKER = priority_ranker
    return _PRIORITY_RANKER

# ---------------------------------------------------------------------------
# ATS platform URL patterns. Anchored to canonical host to avoid swallowing
# unrelated company domains. Each method key is short, lowercase, stable.
# ---------------------------------------------------------------------------
_ATS_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "lever": [
        re.compile(r"https?://jobs\.lever\.co/[^\s)>\]\"'\u200b]+", re.I),
    ],
    "greenhouse": [
        re.compile(r"https?://boards\.greenhouse\.io/[^\s)>\]\"'\u200b]+", re.I),
        re.compile(r"https?://job-boards\.greenhouse\.io/[^\s)>\]\"'\u200b]+", re.I),
        re.compile(r"https?://[a-z0-9-]+\.greenhouse\.io/[^\s)>\]\"'\u200b]+", re.I),
    ],
    "ashby": [
        re.compile(r"https?://jobs\.ashbyhq\.com/[^\s)>\]\"'\u200b]+", re.I),
        re.compile(r"https?://[a-z0-9-]+\.ashbyhq\.com/[^\s)>\]\"'\u200b]+", re.I),
    ],
    "workday": [
        re.compile(
            r"https?://[a-z0-9-]+(?:\.wd\d+)?\.myworkdayjobs\.com/[^\s)>\]\"'\u200b]+",
            re.I,
        ),
    ],
    "workable": [
        re.compile(r"https?://apply\.workable\.com/[^\s)>\]\"'\u200b]+", re.I),
        re.compile(r"https?://[a-z0-9-]+\.workable\.com/[^\s)>\]\"'\u200b]+", re.I),
    ],
    "smartrecruiters": [
        re.compile(r"https?://jobs\.smartrecruiters\.com/[^\s)>\]\"'\u200b]+", re.I),
        re.compile(r"https?://careers\.smartrecruiters\.com/[^\s)>\]\"'\u200b]+", re.I),
    ],
    "recruitee": [
        re.compile(r"https?://[a-z0-9-]+\.recruitee\.com/[^\s)>\]\"'\u200b]+", re.I),
    ],
    "teamtailor": [
        re.compile(r"https?://[a-z0-9-]+\.teamtailor\.com/[^\s)>\]\"'\u200b]+", re.I),
    ],
    "jobvite": [
        re.compile(r"https?://jobs\.jobvite\.com/[^\s)>\]\"'\u200b]+", re.I),
    ],
    "icims": [
        re.compile(r"https?://careers-[a-z0-9-]+\.icims\.com/[^\s)>\]\"'\u200b]+", re.I),
        re.compile(r"https?://[a-z0-9-]+\.icims\.com/[^\s)>\]\"'\u200b]+", re.I),
    ],
    "bamboohr": [
        re.compile(
            r"https?://[a-z0-9-]+\.bamboohr\.(?:com|co\.uk)/[^\s)>\]\"'\u200b]+",
            re.I,
        ),
    ],
    "keka": [
        re.compile(r"https?://[a-z0-9-]+\.keka\.com/[^\s)>\]\"'\u200b]+", re.I),
    ],
    "zoho": [
        re.compile(
            r"https?://[a-z0-9-]+\.zohorecruit\.(?:com|in|eu)/[^\s)>\]\"'\u200b]+",
            re.I,
        ),
    ],
    "wellfound": [
        re.compile(r"https?://(?:www\.)?wellfound\.com/jobs/[^\s)>\]\"'\u200b]+", re.I),
        re.compile(r"https?://(?:www\.)?angel\.co/(?:jobs|company)/[^\s)>\]\"'\u200b]+", re.I),
    ],
    "breezy": [
        re.compile(r"https?://[a-z0-9-]+\.breezy\.hr/[^\s)>\]\"'\u200b]+", re.I),
    ],
    "dover": [
        re.compile(r"https?://(?:app|www)\.dover\.com/(?:jobs|apply)[^\s)>\]\"'\u200b]*", re.I),
    ],
    "naukri": [
        re.compile(r"https?://(?:www\.)?naukri\.com/job-listings[^\s)>\]\"'\u200b]+", re.I),
    ],
    "instahyre": [
        re.compile(r"https?://(?:www\.)?instahyre\.com/(?:job|apply)[^\s)>\]\"'\u200b]+", re.I),
    ],
    "cutshort": [
        re.compile(r"https?://(?:www\.)?cutshort\.io/[^\s)>\]\"'\u200b]+", re.I),
    ],
}

# LinkedIn job permalink (Easy Apply entry point)
_LINKEDIN_JOB_RE = re.compile(
    r"https?://(?:www\.)?linkedin\.com/jobs/view/\d+[^\s)>\]\"'\u200b]*",
    re.I,
)

# Explicit easy-apply phrasing (often used without a URL in the body)
_EASY_APPLY_TEXT_RES = [
    re.compile(r"\beasy\s*apply\b", re.I),
    re.compile(r"\bapply\s+on\s+linkedin\b", re.I),
    re.compile(r"\bapply\s+via\s+linkedin\b", re.I),
    re.compile(r"\bapply\s+through\s+linkedin\b", re.I),
]

# General email detector. Strict enough to require a real TLD (≥2 chars) and
# not starting/ending with a dot or hyphen in the host label.
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+"
    r"\b"
)

# Mail-intent phrases. We require a phrase AND an email in the post for the
# "mail" method to fire — this avoids counting random @-mentions or footer
# signatures as a hiring application channel.
_MAIL_INTENT_RES = [
    re.compile(r"\bmail\s+(?:us|me|at|to|your|it|them)\b", re.I),
    re.compile(r"\bemail\s+(?:us|me|at|to|your|it|them)\b", re.I),
    re.compile(r"\bdrop(?:ping)?\s+(?:a|an|your)?\s*(?:mail|cv|resume|note|line|email)\b", re.I),
    re.compile(
        r"\bsend\s+(?:your\s+|a\s+|an\s+)?"
        r"(?:cv|cvs|resume|resumes|profile|profiles|mail|email|application|applications|portfolio|details|bio)\b",
        re.I,
    ),
    re.compile(
        r"\bshare\s+(?:your\s+|a\s+)?"
        r"(?:cv|resume|profile|portfolio|details|bio|experience)\b",
        re.I,
    ),
    re.compile(r"\bapply\s+(?:via|by|through|over)\s+(?:mail|e?mail)\b", re.I),
    re.compile(r"\bwrite\s+(?:to|at)\s+(?:us|me)\b", re.I),
    re.compile(r"\breach\s+(?:out\s+)?(?:to|at)\s+(?:us|me)\b", re.I),
    re.compile(r"\bshoot\s+(?:a|an)?\s*(?:mail|email|cv|resume)\b", re.I),
    re.compile(r"\bforward\s+(?:your\s+|a\s+)?(?:cv|resume|profile|application)\b", re.I),
]

# Email local-parts that indicate automated senders, never a hiring channel.
# Require mail-intent phrase AND a non-blocklisted email — emails with these
# local-parts are excluded from the `apply_emails` list.
_EMAIL_LOCAL_BLOCKLIST = {
    "noreply", "no-reply", "donotreply", "do-not-reply",
    "mailer-daemon", "postmaster", "bounces",
    "privacy", "legal", "abuse", "unsubscribe",
}

# ---------------------------------------------------------------------------
# Soft-apply (DM / comment-interested) channel
# ---------------------------------------------------------------------------
#
# Many legit recruiter posts carry no URL and no email — they say "DM me
# your resume", "comment 'interested' and I'll reach out", "ping me with
# your CV". The default gate above rejects those outright (no gform, no
# ATS, no email) even though they are unambiguously hiring posts.
#
# The soft-apply gate is OPT-IN-but-on-by-default and two-gated:
#
#   Signal 1 (soft-apply language) — one of the patterns below must fire
#   in body or comments. This covers the DM / comment-interested / drop-
#   resume / ping-me archetype.
#
#   Signal 2 (hiring intent) — the post MUST also pass either
#   `is_hiring_intent_author` (author headline is Recruiter/HR/Founder/…)
#   OR `has_recruiter_language` (body reads like a recruiter post:
#   "we're hiring 3 PMs", `#nowhiring`, etc.).
#
# Both gates → `apply_methods = ["soft_dm"]`. This keeps recall high on
# real recruiter posts while avoiding false positives on generic content
# that merely happens to say "DM me your thoughts on X".
_SOFT_APPLY_RES = [
    # Direct "DM me" / "PM me" / "message me" / "inbox me"
    re.compile(r"\b(?:dm|pm|ping|message|inbox|text|msg|whatsapp|wa)\s+me\b", re.I),
    # "Send a DM" / "drop me a DM" / "send me a message"
    re.compile(r"\b(?:send|drop|shoot)\s+(?:me\s+)?(?:a|an)\s*(?:dm|pm|message|ping|msg|line|note|email|mail)\b", re.I),
    # "Reach out to me" / "hit me up"
    re.compile(r"\breach\s+(?:out\s+)?to\s+me\b", re.I),
    re.compile(r"\bhit\s+me\s+up\b", re.I),
    # "Drop your resume/CV in comments/DM/inbox"
    re.compile(r"\bdrop\s+(?:your\s+|a\s+)?(?:resume|cv|profile|portfolio|details)\s+(?:in|at|on|via)?\s*(?:the\s+)?(?:comment|comments|dm|pm|inbox|message)\b", re.I),
    # "Share your profile / CV in comments / DM / below"
    re.compile(r"\bshare\s+(?:your\s+|a\s+)?(?:resume|cv|profile|details|experience|background|interest)(?:\s+(?:in|via|over|through|on))?\s+(?:the\s+)?(?:comment|comments|dm|pm|below|below\s+in\s+comments|message|inbox)?\b", re.I),
    # "Comment 'interested'" / "drop 'interested' below" / "type 'interested'"
    re.compile(r"\b(?:comment|drop|type|say|write|post|put|reply|add)\s+[\"']?(?:interested|me|count\s+me\s+in|in|keen|pick\s+me)[\"']?\b", re.I),
    # "Leave a comment below with …"
    re.compile(r"\bleave\s+(?:a\s+)?(?:comment|reply)\s+(?:below\s+)?(?:with|if\s+you)\b", re.I),
    # "Tag someone you know"
    re.compile(r"\btag\s+(?:someone|a\s+friend|anyone|people|folks)\s+(?:who|you\s+know)", re.I),
    # "Let me know" / "slide into my DMs"
    re.compile(r"\blet\s+me\s+know\s+(?:in\s+(?:the\s+)?comments|if\s+(?:you|interested)|below)\b", re.I),
    re.compile(r"\bslide\s+into\s+my\s+dms?\b", re.I),
    # "Who wants / who's interested" — opening for a casual-recruitment post
    re.compile(r"\bwho(?:'s|\s+is)\s+(?:interested|keen|up\s+for|looking)\b", re.I),
]


# ---------------------------------------------------------------------------
# Text collection — body + top-level comments + nested replies
# ---------------------------------------------------------------------------

def _walk_comment_texts(comments: list[Any]) -> list[str]:
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


def _collect_texts(post: dict[str, Any]) -> list[str]:
    texts = [post.get("text") or ""]
    texts.extend(_walk_comment_texts(post.get("comments") or []))
    return [t for t in texts if t]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim_trailing(url: str) -> str:
    while url and url[-1] in ".,);]}>\"'\u200b":
        url = url[:-1]
    return url


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


# ---------------------------------------------------------------------------
# Per-signal detectors
# ---------------------------------------------------------------------------

def _find_ats_urls(texts: list[str]) -> dict[str, list[str]]:
    """Return {method_name: [canonical_urls, ...]}."""
    hits: dict[str, list[str]] = {}
    for t in texts:
        for method, patterns in _ATS_PATTERNS.items():
            for p in patterns:
                for m in p.findall(t):
                    url = _trim_trailing(m)
                    if not url:
                        continue
                    bucket = hits.setdefault(method, [])
                    if url not in bucket:
                        bucket.append(url)
    return hits


def _find_linkedin_easy_apply(texts: list[str]) -> tuple[bool, list[str]]:
    """Detect either a linkedin.com/jobs/view URL or an explicit 'easy apply' phrase."""
    urls: list[str] = []
    phrase_hit = False
    for t in texts:
        for m in _LINKEDIN_JOB_RE.findall(t):
            u = _trim_trailing(m)
            if u and u not in urls:
                urls.append(u)
        if not phrase_hit:
            phrase_hit = any(rx.search(t) for rx in _EASY_APPLY_TEXT_RES)
    return (bool(urls) or phrase_hit), urls


def _find_mail_to_apply(texts: list[str]) -> tuple[bool, list[str]]:
    """Return (detected, cleaned_emails). Detected only when BOTH an email AND a
    mail-intent phrase appear somewhere in the post text."""
    emails: list[str] = []
    has_intent = False
    for t in texts:
        for match in _EMAIL_RE.findall(t):
            email = match.lower()
            local = email.split("@", 1)[0]
            if local in _EMAIL_LOCAL_BLOCKLIST:
                continue
            if email not in emails:
                emails.append(email)
        if not has_intent:
            has_intent = any(rx.search(t) for rx in _MAIL_INTENT_RES)
    detected = bool(emails) and has_intent
    return detected, emails


def _find_soft_dm(post: dict[str, Any], texts: list[str]) -> bool:
    """Detect a soft-apply (DM / comment-interested) channel on a LinkedIn post.

    Two-gated: requires (a) one of the `_SOFT_APPLY_RES` patterns in body or
    a top-level comment AND (b) a hiring-intent signal from priority_ranker
    (author is recruiter/HR/founder OR body reads like a recruiter post).

    LinkedIn posts only — job-board sources always carry a structured URL
    so soft-apply is never needed for them and firing it would muddy the
    analytics. Feature-flagged via `config.ENABLE_SOFT_APPLY`.
    """
    if not config.ENABLE_SOFT_APPLY:
        return False
    if post.get("source") != "linkedin_post":
        return False

    # Gate 1: soft-apply language
    lang_hit = any(
        any(rx.search(t) for rx in _SOFT_APPLY_RES) for t in texts
    )
    if not lang_hit:
        return False

    # Gate 2: corroborating hiring signal
    pr = _priority_ranker()
    return pr.is_hiring_intent_author(post) or pr.has_recruiter_language(post)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Sources whose records arrive with a pre-known structured apply URL —
# LinkedIn's job board, Naukri, Indeed, etc. For these we skip free-text
# detection entirely and treat the record as already-actionable. Saves
# compute, and more importantly prevents false negatives when a job
# description happens to lack the literal word "apply" (it's already a
# job listing — the apply is implied).
_STRUCTURED_JOB_SOURCES = {"linkedin_job", "naukri", "indeed"}


def detect_apply_methods(post: dict[str, Any]) -> dict[str, Any]:
    """Return a dict of apply-signal annotations for a single post.

    Always returns the four keys (empty lists when a signal didn't fire), so
    callers can merge unconditionally: `{**post, **detect_apply_methods(post)}`.

    Records from structured job-board sources (LinkedIn jobs, Naukri, Indeed)
    short-circuit to `apply_methods=[source]` and `apply_urls=[apply_url]` —
    the listing IS the apply channel.
    """
    source = post.get("source")
    if source in _STRUCTURED_JOB_SOURCES:
        # Prefer the pre-computed apply_url; fall back to post_url; never empty.
        apply_url = post.get("apply_url") or post.get("post_url") or ""
        urls = [apply_url] if apply_url else []
        return {
            "apply_methods": [source],
            "apply_urls": urls,
            "apply_emails": [],
            "gform_urls": [],
        }

    texts = _collect_texts(post)

    methods: list[str] = []
    urls: list[str] = []

    # 1. Google Form — use the existing detector (handles shorteners, nested replies)
    gforms = gform_filter.find_gform_links(post)
    if gforms:
        methods.append("gform")
        urls.extend(gforms)

    # 2. ATS platforms
    for method, ats_urls in _find_ats_urls(texts).items():
        methods.append(method)
        urls.extend(ats_urls)

    # 3. LinkedIn Easy Apply
    ea_detected, ea_urls = _find_linkedin_easy_apply(texts)
    if ea_detected:
        methods.append("linkedin_easy_apply")
        urls.extend(ea_urls)

    # 4. Mail-to-apply
    mail_detected, emails = _find_mail_to_apply(texts)
    if mail_detected:
        methods.append("mail")

    # 5. Soft-apply (DM / comment-interested). Fires ONLY when gated hiring
    #    signals also fire — see `_find_soft_dm`. This recovers legitimate
    #    recruiter posts that have no URL or email at all.
    if _find_soft_dm(post, texts):
        methods.append("soft_dm")

    return {
        "apply_methods": methods,
        "apply_urls": _dedupe_preserve(urls),
        "apply_emails": emails if mail_detected else [],
        "gform_urls": gforms,
    }


def filter_posts_with_apply_method(
    posts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only posts that expose at least one machine-actionable apply method.

    Each kept post is returned with the annotations from `detect_apply_methods`
    merged in. Ordering of inputs is preserved.
    """
    kept: list[dict[str, Any]] = []
    for p in posts:
        info = detect_apply_methods(p)
        if info["apply_methods"]:
            kept.append({**p, **info})
    return kept


def method_distribution(posts: list[dict[str, Any]]) -> dict[str, int]:
    """Summarize a list of annotated posts by apply method. A single post can
    contribute to multiple buckets (e.g. both `gform` and `mail`)."""
    counts: dict[str, int] = {}
    for p in posts:
        for m in p.get("apply_methods") or []:
            counts[m] = counts.get(m, 0) + 1
    return counts
