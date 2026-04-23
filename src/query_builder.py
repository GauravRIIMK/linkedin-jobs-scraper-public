"""Build LinkedIn search queries per track from saved role preferences.

Design principles (tuned after a dry-run returning 0 posts with exact-title queries):

1. **Cities are NEVER a hard filter.**  Recruiters rarely include the city in a
   hiring post's searchable text; the city appears in a banner or the Jobs
   metadata. Adding `Bangalore` to a search query is a precision trick that
   destroys recall. Location fit is scored by the Haiku reranker, which sees
   the resume summary and the full JD.

2. **Two versions of every role.**  For "Growth Marketing Manager" we derive:
      specific = "Growth Marketing"    (drop trailing generic suffix "Manager")
      broad    = "Growth"              (first non-stopword token)
   We use `specific` as the high-precision driver and fall back to `broad` for
   a safety-net firehose query.

3. **Six-tier query ladder, not a cartesian product.**  Each tier adds a
   different kind of recall:
      T1  #hiring "<specific>"              — precision, top 5 roles
      T2  #hiring <specific> India          — India scope, top 2 roles
      T3  "we are hiring" <specific[0]>     — non-hashtag phrasing
      T4  #hiring "<specific>" remote       — remote variant (if opted in)
      T5  #hiring <broad[0]> India          — safety-net firehose
      T6  #hiring "<synonym>"               — paraphrase catchment (top 2 × 3)

4. **Respect `QUERIES_PER_TRACK`.**  Later tiers are pruned if the cap is hit
   before they're emitted. T1 always wins because it yields the best signal.
"""
from __future__ import annotations

import re
from typing import Any

from . import config

# Seniority prefixes that contain no role information on their own.
# Stored lowercase with a trailing dot tolerated.
_SENIORITY_PREFIXES = {
    "senior", "sr", "junior", "jr",
    "associate", "assistant", "principal",
}

# Trailing title suffixes that don't carry role specificity.
# Intentionally narrow — "Director", "Lead", "Consultant" keep the role intact.
_GENERIC_SUFFIX = {"manager", "mgr"}

# Tokens that are too broad to be the single-word "broad" fallback. Titles,
# stopwords, prepositions. Acronyms (AI, ML, UX) are explicitly NOT here.
_BAD_BROADS = {
    "the", "a", "an", "and", "for", "of", "to", "with", "by", "in", "at", "on",
    "head", "lead", "chief", "director", "vp", "staff", "senior", "junior",
    "manager", "mgr", "principal", "associate", "officer",
}

# Role-synonym catchment. Key is the LOWER-CASED specific phrase (after
# _role_specific() strip); values are 2-4 paraphrases that a recruiter might
# use in a #hiring post instead of the canonical title.
#
# This is a RECALL tool — the reranker still scores every candidate, so wrong
# matches land in the top-K then get dropped by Haiku. The goal here is to
# stop missing posts that use a synonym the resume summary doesn't say.
#
# Synonyms are intentionally SHORT — LinkedIn post search is bag-of-words;
# longer phrases don't add precision, they just narrow recall.
_ROLE_SYNONYMS: dict[str, list[str]] = {
    # Growth / performance marketing
    "growth marketing": ["growth manager", "performance marketing", "growth hacker", "performance marketer"],
    "growth": ["growth manager", "growth marketing", "performance marketing"],
    "performance marketing": ["growth marketing", "paid media", "paid acquisition"],
    "digital marketing": ["performance marketing", "growth marketing", "online marketing"],
    "marketing": ["brand marketing", "growth marketing", "digital marketing"],
    "product marketing": ["pmm", "go-to-market", "gtm"],
    # Product management
    "product management": ["product manager", "pm", "associate product manager"],
    "product": ["product manager", "technical product manager", "tpm"],
    "technical product": ["technical product manager", "tpm", "platform pm"],
    # Data / AI
    "data science": ["data scientist", "ml engineer", "applied scientist"],
    "data analytics": ["data analyst", "analytics", "business analyst"],
    "data": ["data scientist", "data analyst", "analytics"],
    "machine learning": ["ml engineer", "applied ai", "ai engineer"],
    "ai": ["ai engineer", "applied ai", "genai engineer"],
    # Engineering
    "software": ["software engineer", "sde", "full stack"],
    "full stack": ["fullstack engineer", "sde", "software engineer"],
    "backend": ["backend engineer", "sde", "server engineer"],
    "frontend": ["frontend engineer", "web engineer", "ui engineer"],
    "devops": ["site reliability", "platform engineer", "sre"],
    "platform": ["platform engineer", "devops", "sre"],
    # Strategy / ops
    "strategy": ["strategy & operations", "bizops", "business operations"],
    "operations": ["ops manager", "bizops", "strategy & operations"],
    "business development": ["bizdev", "bd", "partnerships"],
    # Design
    "product design": ["product designer", "ui ux", "ux designer"],
    "ux": ["product designer", "ui ux designer", "ux researcher"],
    # Program / project
    "program management": ["program manager", "tpm", "technical program manager"],
    "project management": ["project manager", "delivery manager"],
}


def _tokens(role: str) -> list[str]:
    return [t for t in re.split(r"\s+", (role or "").strip()) if t]


def _norm(word: str) -> str:
    return word.lower().rstrip(".")


def _role_specific(role: str) -> str:
    """Strip leading seniority + trailing generic suffix. Falls back to the
    original role if stripping would leave nothing useful."""
    toks = _tokens(role)
    if not toks:
        return role
    # Strip a single leading seniority token.
    if _norm(toks[0]) in _SENIORITY_PREFIXES:
        toks = toks[1:]
    # Strip trailing generic suffix only if meaningful tokens remain.
    if len(toks) > 1 and _norm(toks[-1]) in _GENERIC_SUFFIX:
        toks = toks[:-1]
    if not toks:
        return role
    return " ".join(toks)


def _role_broad(role: str) -> str:
    """Return the single most-specific keyword from a role title.

    Scans the specific form left-to-right and picks the first token that is
    (a) more than 1 character and (b) not in `_BAD_BROADS`. Falls back to the
    full specific phrase if no token qualifies (e.g. "Chief of Staff")."""
    specific = _role_specific(role)
    toks = _tokens(specific)
    for t in toks:
        if len(t) > 1 and _norm(t) not in _BAD_BROADS:
            return t
    return specific


def _synonyms_for(specific: str) -> list[str]:
    """Return alternate phrases to search for. Keyed by lowercased specific
    phrase. Falls back to the lowercased broad token if no phrase match."""
    key = _norm(specific)
    if key in _ROLE_SYNONYMS:
        return _ROLE_SYNONYMS[key]
    # Also try the broad token — e.g. "Growth Marketing" → broad "growth"
    broad_key = _norm(_role_broad(specific))
    if broad_key != key and broad_key in _ROLE_SYNONYMS:
        return _ROLE_SYNONYMS[broad_key]
    return []


def _dedupe_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out


def build_queries_for_track(track_prefs: dict[str, Any]) -> list[str]:
    """Produce LinkedIn search queries for one track. Capped at
    `config.QUERIES_PER_TRACK`. See module docstring for the ladder."""
    roles: list[str] = track_prefs.get("roles", []) or []
    include_remote: bool = track_prefs.get("include_remote", True)
    if not roles:
        return []

    specifics = _dedupe_preserve([_role_specific(r) for r in roles])
    broads = _dedupe_preserve([_role_broad(r) for r in roles])

    queries: list[str] = []
    seen: set[str] = set()

    def add(q: str) -> None:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            queries.append(q)

    # T1 precision: quoted specific + #hiring (top 5)
    for s in specifics[:5]:
        add(f'#hiring "{s}"')

    # T2 India scope: unquoted to let LinkedIn stemmer do its work (top 2)
    for s in specifics[:2]:
        add(f"#hiring {s} India")

    # T3 non-hashtag phrasing — some recruiters skip the #hiring hashtag
    if specifics:
        add(f'"we are hiring" {specifics[0]}')

    # T4 remote variant (only if opted in; top 2 specifics)
    if include_remote:
        for s in specifics[:2]:
            add(f'#hiring "{s}" remote')

    # T5 safety-net firehose: broad keyword + India scope (top 1)
    for b in broads[:1]:
        add(f"#hiring {b} India")

    # T6 synonym catchment: up to 3 synonyms for the top-2 specifics. This is
    # the recall tier — we pay $0.002/post but the reranker drops misfits, so
    # the cost of a miss is much higher than the cost of an extra query.
    for s in specifics[:2]:
        for syn in _synonyms_for(s)[:3]:
            add(f'#hiring "{syn}"')

    return queries[: config.QUERIES_PER_TRACK]


def build_all_queries(preferences: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for track in preferences.get("tracks", []):
        if not track.get("active", True):
            continue
        out[track["track"]] = build_queries_for_track(track)
    return out
