"""Per-track role-archetype gate.

Motivation (direct from observation):
  The `ai_ops_automation` track was surfacing ML engineer / software
  developer / data scientist roles — because those JDs share surface-level
  tokens with the user's track summary ("AI", "Python", "automation",
  "agents"). Voyage embedding puts them in the same neighborhood, Haiku
  gives them a 7/10 based on skill overlap, they slip through.

  But the user self-describes as a **vibecoder, not a coder** — someone who
  uses AI tools to assemble solutions without writing production code. A
  Software Engineer JD that asks for "5+ years Python, distributed systems,
  production ML" is NOT a fit, no matter how many tokens overlap.

Design:
  - Each track can declare a list of TITLE-LEVEL negatives. If a negative
    matches the post's author_title OR the first 240 chars of JD body,
    the post is rejected UNLESS a positive (override) phrase also fires.
  - We check ONLY the "title window" (not the whole body) because most JDs
    put the role title at the top. Deep-body keyword matches are too noisy
    — an SDE JD that mentions "collaborate with product managers" shouldn't
    look like a PM role.
  - Word boundaries via `\b` avoid false positives (e.g. `sde` inside
    `designer`).
  - Other tracks (growth_marketing, product_brand_category, strategy_consulting)
    ship with empty lists — they don't have the coder/operator ambiguity.
    Add negatives there too if similar noise shows up.
"""
from __future__ import annotations

import re
from typing import Any

# ---------------------------- Negatives ----------------------------
# Titles that indicate a traditional-coder role. Any match → reject
# (subject to positives rescue).

TRACK_NEGATIVES: dict[str, list[str]] = {
    "ai_ops_automation": [
        # Core SWE titles
        "software engineer",
        "software developer",
        "software dev",
        "sde",
        "swe",
        "sde-1",
        "sde-2",
        "sde-3",
        "sde i",
        "sde ii",
        "sde iii",
        "sde 1",
        "sde 2",
        "sde 3",
        "backend engineer",
        "backend developer",
        "back-end engineer",
        "back-end developer",
        "frontend engineer",
        "frontend developer",
        "front-end engineer",
        "front-end developer",
        "full stack engineer",
        "full-stack engineer",
        "fullstack engineer",
        "full stack developer",
        "full-stack developer",
        "fullstack developer",
        "mobile engineer",
        "android developer",
        "android engineer",
        "ios developer",
        "ios engineer",
        # ML / DS / Research titles (deep coding / math)
        "ml engineer",
        "machine learning engineer",
        "mlops engineer",
        "data scientist",
        "data engineer",
        "nlp engineer",
        "ai researcher",
        "research engineer",
        "research scientist",
        "computer vision engineer",
        "applied scientist",
        "deep learning engineer",
        # Infra / platform / sec (deep coding)
        "devops engineer",
        "site reliability",
        "site reliability engineer",
        "sre",
        "platform engineer",
        "infrastructure engineer",
        "security engineer",
        "cloud engineer",
        "qa engineer",
        "sdet",
        "automation test engineer",
        "test engineer",
        # Language-specific (strong coder signal)
        "python developer",
        "java developer",
        "node.js developer",
        "nodejs developer",
        "node developer",
        "golang developer",
        "go developer",
        "rust developer",
        "react developer",
        "angular developer",
        "php developer",
        "ruby developer",
        "c++ developer",
        # Senior engineering ICs (deep coding)
        "principal engineer",
        "staff engineer",
        "distinguished engineer",
    ],
    # ----------------------------------------------------------------
    # Marketing track: user has NO experience in digital marketing or
    # performance marketing (explicit ask: "strictly avoid"). The negatives
    # target the title-level phrases used by job-boards for those sub-
    # specializations. Rescued by the positives list below when a broader
    # role (growth / brand / content / product marketing) just happens to
    # mention performance or digital as one capability.
    #
    # Note: "paid ads", "paid media", "paid acquisition", "google ads",
    # "meta ads", "facebook ads", "SEO/SEM specialist" are treated as the
    # same archetype since they sit inside the digital/performance family.
    # ----------------------------------------------------------------
    "growth_marketing": [
        # Core digital / performance titles
        "digital marketing",
        "digital marketer",
        "digital marketing specialist",
        "digital marketing executive",
        "digital marketing manager",
        "digital marketing lead",
        "digital marketing head",
        "performance marketing",
        "performance marketer",
        "performance marketing specialist",
        "performance marketing executive",
        "performance marketing manager",
        "performance marketing lead",
        "performance marketing head",
        # Sub-specializations within digital / performance
        "paid media",
        "paid ads",
        "paid acquisition",
        "paid marketing",
        "paid social",
        "paid search",
        "media buyer",
        "media planner",
        "media buying",
        "ppc specialist",
        "ppc manager",
        "google ads specialist",
        "google ads manager",
        "meta ads specialist",
        "meta ads manager",
        "facebook ads specialist",
        "facebook ads manager",
        "sem specialist",
        "sem manager",
        "sem executive",
        "seo specialist",
        "seo executive",
        "seo manager",
        "seo lead",
        "seo head",
        "seo analyst",
        "programmatic specialist",
        "programmatic manager",
        "ad operations",
        "ad ops",
        "adwords specialist",
    ],
}

# ---------------------------- Positives (rescues) ----------------------------
# If any of these phrases appears alongside a negative, KEEP the post. These
# are the exact builder/operator/strategy archetypes the user wants. Kept
# short to minimize false-rescues.

TRACK_POSITIVES: dict[str, list[str]] = {
    "ai_ops_automation": [
        "ai product manager",
        "ai pm",
        "ai program manager",
        "ai operator",
        "ai builder",
        "ai transformation",
        "ai strategist",
        "ai consultant",
        "ai enablement",
        "ai adoption",
        "ai orchestration",
        "ai implementation",
        "ai integration lead",
        "ai ops",
        "aiops",
        "automation specialist",
        "automation lead",
        "automation manager",
        "head of automation",
        "head of ai",
        "director of automation",
        "director of ai",
        "chief of staff",
        "workflow automation",
        "business automation",
        "process automation",
        "no-code builder",
        "low-code builder",
        "agent builder",
        "solutions architect",
    ],
    # ----------------------------------------------------------------
    # Growth-marketing positives. If one of these appears in the title
    # window, a `digital marketing` / `performance marketing` negative is
    # rescued — that's a sign the primary role is growth/brand/content/
    # product marketing, with digital/performance listed only as a
    # supporting skill. The user explicitly wants these kept.
    #
    # KEPT INTENTIONALLY TIGHT:
    #   Every positive must DISCRIMINATE between the digital/performance
    #   family and other sub-specializations. Generic phrases like
    #   "marketing manager" / "head of marketing" / "marketing director"
    #   do NOT belong here — they match the very titles we want to reject
    #   ("Digital Marketing Manager", "Head of Digital Marketing", …) and
    #   would rescue them by accident. Plain unlabeled titles like "Head
    #   of Marketing" already hit zero negatives so they pass without
    #   needing a positive rescue; the positives only exist to save mixed
    #   roles where a growth/brand/content/product signal sits alongside
    #   a digital/performance mention.
    # ----------------------------------------------------------------
    "growth_marketing": [
        # Growth-specific
        "growth marketing",
        "growth marketer",
        "head of growth",
        "growth lead",
        "growth manager",
        "growth hacker",
        "growth strategist",
        # Brand-specific
        "brand marketing",
        "brand marketer",
        "brand manager",
        "brand strategist",
        # Product-marketing-specific
        "product marketing",
        "product marketer",
        "pmm",
        # Content-specific (the word "content" + marketing role words,
        # never just a bare title like "Content Manager" which is fine
        # on its own and doesn't need rescuing).
        "content marketing",
        "content marketer",
        "content strategist",
        "content manager",
        "content lead",
        # Lifecycle / retention / CRM — all orthogonal to paid/digital
        "lifecycle marketing",
        "retention marketing",
        "crm marketing",
        # Category / integrated / consumer — distinct archetypes
        "category marketing",
        "category manager",
        "integrated marketing",
        "consumer marketing",
        "consumer insights",
    ],
}


def _compile_phrase(phrase: str) -> re.Pattern[str]:
    """Compile a multi-word phrase into a whole-word, hyphen/space-tolerant regex.

    Tokens from the phrase are re.escape'd and joined by `[\\s\\-]+`, so the
    same pattern matches "full-stack developer", "full stack developer", and
    "fullstack-developer". Wrapped in `\\b` word-boundaries to prevent
    substring matches (e.g. `sde` inside `designer`).
    """
    tokens = [t for t in re.split(r"[\s\-]+", phrase.strip()) if t]
    body = r"[\s\-]+".join(re.escape(t) for t in tokens)
    return re.compile(rf"\b{body}\b", re.I)


_NEG_CACHE: dict[str, list[re.Pattern[str]]] = {}
_POS_CACHE: dict[str, list[re.Pattern[str]]] = {}


def _negatives_for(track: str) -> list[re.Pattern[str]]:
    if track not in _NEG_CACHE:
        _NEG_CACHE[track] = [_compile_phrase(p) for p in TRACK_NEGATIVES.get(track, [])]
    return _NEG_CACHE[track]


def _positives_for(track: str) -> list[re.Pattern[str]]:
    if track not in _POS_CACHE:
        _POS_CACHE[track] = [_compile_phrase(p) for p in TRACK_POSITIVES.get(track, [])]
    return _POS_CACHE[track]


# Remove JD-extractor framing so `[post body]` / `[from comments]` etc. don't
# distort word-boundary checks.
_FRAMING_RE = re.compile(r"\[(?:post body|from comments|from https?://[^\]]+)\]", re.I)


def _title_window(post: dict[str, Any]) -> str:
    """Text we scan for title-like signals.

    Starts with the most reliable signals (structured title/headline fields),
    then appends the first 480 chars of JD body. Widened from 240 → 480
    because recruiter posts on LinkedIn often lead with 2-3 sentences of
    company context before naming the role, so the role title can sit at
    char ~300-400. We accepted the extra false-positive risk because a
    negative match still requires a positive-rescue to be lifted.
    """
    parts: list[str] = []
    for key in ("title", "job_title", "role", "author_title", "headline"):
        v = post.get(key)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    body = post.get("jd_text") or post.get("text") or ""
    body = _FRAMING_RE.sub(" ", body)[:480]
    if body:
        parts.append(body)
    return "\n".join(parts)


# Patterns that declare a role later in the body — covers JDs whose
# headline is generic ("We're hiring!") but the role title sits at char
# 500-2000. These bracket the negative phrase, so "{verb} {negative}" is
# treated as a title-level hit. Scans up to 2000 chars to catch
# mid-paragraph declarations without loading the entire JD into regex.
_ROLE_DECLARATIVE_PREFIXES = (
    r"(?:we\s+are\s+|we're\s+|we\s+)?(?:hiring|recruiting|looking\s+for|seeking|on(?:\s+the)?\s+lookout\s+for)"
    r"|role\s*[:\-–]|position\s*[:\-–]|job\s*title\s*[:\-–]|title\s*[:\-–]"
    r"|open\s+(?:role|position)s?\s*[:\-–]?"
    r"|we\s+need(?:\s+a)?"
)
_SENIORITY_INTERLUDE = r"(?:\s+(?:a|an|the|senior|sr\.?|junior|jr\.?|lead|head\s+of|principal|staff|associate|assistant))*"


def _body_declares_negative(track: str, body: str) -> bool:
    """True iff a negative phrase appears after a role-declarative preamble
    anywhere in the first ~2000 chars of body. Complements the title-window
    check for long preamble posts."""
    negatives_raw = TRACK_NEGATIVES.get(track) or []
    if not negatives_raw or not body:
        return False
    scan = body[:2000]
    for phrase in negatives_raw:
        # Reuse _compile_phrase's token rules so "digital marketing" also
        # matches "digital-marketing" and "digital  marketing".
        tokens = [t for t in re.split(r"[\s\-]+", phrase.strip()) if t]
        if not tokens:
            continue
        neg_body = r"[\s\-]+".join(re.escape(t) for t in tokens)
        pat = re.compile(
            rf"\b(?:{_ROLE_DECLARATIVE_PREFIXES}){_SENIORITY_INTERLUDE}\s+{neg_body}\b",
            re.I,
        )
        if pat.search(scan):
            return True
    return False


def passes_role_filter(track: str, post: dict[str, Any]) -> bool:
    """True iff no negative phrase hits, OR a positive overrides a negative hit."""
    negatives = _negatives_for(track)
    if not negatives:
        return True  # no rule for this track → unconditionally pass
    window = _title_window(post)
    title_hit = any(p.search(window) for p in negatives)
    # Also scan the mid-body for role-declarative negatives (covers long
    # recruiter preambles like "hey network! at Acme we're building … we're
    # hiring a Digital Marketing Manager").
    body = post.get("jd_text") or post.get("text") or ""
    body_hit = _body_declares_negative(track, body) if not title_hit else False
    if not (title_hit or body_hit):
        return True  # no negative match → pass
    # Hit a negative — try to rescue with a positive (same window).
    positives = _positives_for(track)
    return any(p.search(window) for p in positives)


def filter_by_role(
    track: str,
    posts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Return (kept, rejected_count). Order preserved within kept."""
    kept = [p for p in posts if passes_role_filter(track, p)]
    return kept, len(posts) - len(kept)
