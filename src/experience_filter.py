"""Experience-level gate: reject JDs whose minimum YoE ask is above the user.

Why a hard filter and not just a reranker hint:
  - The LLM rerank prompt currently says "penalize seniority mismatches" but
    Haiku at haiku-4-5 routinely rates "10+ years of experience required"
    roles at 7/10 anyway when other signals match — because the "fit" score
    is dominated by skill overlap, which tends to be high. A separate hard
    gate before rerank spend saves API cost AND unambiguously drops roles
    that cannot be a fit on seniority alone.
  - Voyage embeddings see "10+ years" as a few tokens in a 500-token JD —
     they don't concentrate the seniority signal enough to pull the
     similarity below threshold.

The filter is conservative: posts with NO detectable YoE requirement pass
unconditionally (we don't drop a role just because the JD happens not to
mention a year count — many JDs describe responsibilities without stating
a minimum).

Patterns adapted from stapply-ai/ats-scrapers' `extract_salary_experience.py`
reference implementation, extended for Indian phrasing ("yrs exp", "min 8
yrs", "8-12 years"). Word boundaries prevent matches inside dates/IDs.
"""
from __future__ import annotations

import re
from typing import Any

from . import config

# ----------------------------- YoE patterns -----------------------------
#
# We accept a match iff the number is followed (within ~20 chars) by a
# year-unit token (year/yr/yoe) — this avoids false positives on "10%
# growth" style numbers. We ALSO require that the phrase sits near a
# "role-requirement" context word (experience/exp/required/minimum/etc.)
# so a JD saying "2+ years of Python" (skill requirement) doesn't bump
# the ceiling to 2 for a role that's actually senior.
#
# Capture group 1 is always the MINIMUM YoE. For ranges like "8-12 yrs"
# group 1 is 8 (the floor — the binding requirement).

_YOE_UNIT = r"(?:years?|yrs?|yoe)"
_ROLE_CTX = (
    r"(?:experience|exp\b|background|work\s+ex|work[- ]ex\b|"
    r"minimum|min\.?\s|at\s*least|atleast|overall|relevant|progressive|"
    r"professional|hands[\-\s]?on|proven|demonstrated|required|must\s+have|"
    r"mandatory|requisite|preferred|industry|career|in\s+(?:the\s+)?role|"
    r"senior|director|head|lead|manager|principal|staff|vp|chief)"
)

_YOE_PATTERNS: list[re.Pattern[str]] = [
    # IMPORTANT: "8-12 years" — range goes FIRST so it wins the non-overlap
    # selection over patterns that'd otherwise capture the upper number
    # (e.g. P6 would see "12 years of experience" inside "8-12 years of
    # experience" and return 12). Range captures the floor (entry bar).
    re.compile(
        rf"\b(\d{{1,2}})\s*(?:-|to|–|—|\bto\b)\s*\d{{1,2}}\s*\+?\s*{_YOE_UNIT}\b",
        re.I,
    ),
    # "10+ years of experience" / "10+ years experience" / "10+ yrs exp"
    re.compile(
        rf"\b(\d{{1,2}})\s*\+\s*{_YOE_UNIT}\s+(?:of\s+)?{_ROLE_CTX}",
        re.I,
    ),
    # "10+ years in {role-context}"
    re.compile(
        rf"\b(\d{{1,2}})\s*\+\s*{_YOE_UNIT}\s+(?:in|of)\s+[\w\s,/-]{{0,60}}{_ROLE_CTX}",
        re.I,
    ),
    # "minimum 8 years" / "at least 8 yrs" / "min. 8 years" / "atleast 8 yrs"
    re.compile(
        rf"\b(?:minimum|min\.?|at\s*least|atleast)\s+(?:of\s+)?(\d{{1,2}})\s*\+?\s*{_YOE_UNIT}\b",
        re.I,
    ),
    # "experience: 8+ years" / "exp: 8 years" (colon-prefixed forms)
    re.compile(
        rf"\b{_ROLE_CTX}\s*[:\-–]\s*(\d{{1,2}})\s*\+?\s*{_YOE_UNIT}\b",
        re.I,
    ),
    # "8+ yrs of {role-context}" / "8 yrs of experience"
    re.compile(
        rf"\b(\d{{1,2}})\s*\+?\s*{_YOE_UNIT}\s+of\s+[\w\s,/-]{{0,30}}{_ROLE_CTX}",
        re.I,
    ),
    # Bare "N+ years" at a sentence boundary (end-of-string or trailing
    # punctuation) with no skill qualifier following — treated as
    # unambiguous seniority signal. Does NOT match "2+ years of Python"
    # because "of Python" follows "years".
    re.compile(
        rf"\b(\d{{1,2}})\s*\+\s*{_YOE_UNIT}\b\s*(?:$|[.,;!?\n])",
        re.I,
    ),
]

# Explicit junior / entry-level signals — if present we KEEP the post even
# if a senior YoE pattern also matched (some JDs say "junior role, up to
# 10 years of team exposure ok"). Rare but worth rescuing.
_JUNIOR_SIGNALS = re.compile(
    r"\b(?:fresher|fresh\s*graduate|entry[\s\-]level|"
    r"0[\s\-]?[\-to]\s*\d\s*(?:years?|yrs?)|"
    r"\bmin(?:imum)?\s+0\s*(?:years?|yrs?)|"
    r"\bjunior\s+(?:role|position|opening))\b",
    re.I,
)


def extract_min_yoe_required(text: str) -> float | None:
    """Return the highest minimum-YoE requirement found, or None.

    Two-stage logic:
      1. Collect every (span, value) match across patterns.
      2. Greedily select non-overlapping spans, earliest-start wins.
         This is what makes "5-10 years of experience" → 5 (the range
         match covers the whole phrase; the inner "10 years of experience"
         singleton is inside it and gets suppressed).
      3. Across the NON-OVERLAPPING set (independent requirements),
         take the MAX — if a JD specifies both "2+ years Python" AND
         "10+ years leadership", the binding constraint is senior one.

    Caveat: this can over-reject when a JD mentions "15 years of industry
    experience" aspirationally while the actual role is mid-level. In
    practice that phrasing is rare; we accept the trade-off because the
    alternative (under-reject) was the status quo bug.
    """
    if not text:
        return None

    candidates: list[tuple[int, int, int]] = []  # (start, end, value)
    for pat in _YOE_PATTERNS:
        for m in pat.finditer(text):
            try:
                v = int(m.group(1))
            except (TypeError, ValueError, IndexError):
                continue
            # Sanity: discard nonsense numbers — >25 is almost always a
            # date/year reference, 0 is entry-level (not a senior-exclusion
            # signal), negatives can't happen but be safe.
            if 1 <= v <= 25:
                candidates.append((m.start(), m.end(), v))

    if not candidates:
        return None

    # Sort by earliest start, then widest span first (prefer the broader
    # range match over a singleton that starts at the same offset).
    candidates.sort(key=lambda c: (c[0], -c[1]))
    selected: list[int] = []
    cursor = -1
    for start, end, v in candidates:
        if start >= cursor:
            selected.append(v)
            cursor = end

    return float(max(selected)) if selected else None


def passes_yoe_filter(
    post: dict[str, Any],
    user_yoe: float | None = None,
    buffer: float | None = None,
) -> bool:
    """True iff the detected-role-minimum is within user's YoE + buffer.

    Absence of a match is PASS (unknown seniority → let LLM handle).
    Junior-signal present is PASS regardless of YoE matches.
    """
    user = user_yoe if user_yoe is not None else config.USER_YOE
    buf = buffer if buffer is not None else config.YOE_BUFFER
    text = post.get("jd_text") or post.get("text") or ""
    if _JUNIOR_SIGNALS.search(text):
        return True
    required = extract_min_yoe_required(text)
    if required is None:
        return True
    return required <= user + buf


def filter_by_yoe(
    posts: list[dict[str, Any]],
    user_yoe: float | None = None,
    buffer: float | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return (kept_posts, rejected_count). Order preserved within kept."""
    kept = [p for p in posts if passes_yoe_filter(p, user_yoe=user_yoe, buffer=buffer)]
    return kept, len(posts) - len(kept)
