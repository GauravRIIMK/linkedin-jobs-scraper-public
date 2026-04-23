"""Post-rerank priority boosts.

Applied AFTER Haiku scoring, BEFORE the final email render. Each signal adds
a configurable boost to `llm_score`; the sum is clamped to [0, 10]. Per-post
`priority_reasons` list is populated so the email can render transparent
badges ("⭐ Tier-1", "🏠 Remote", "💬 Apply in comments", "🔗 LinkedIn post").

Signal taxonomy (ordered by reliability, not by boost size):

  1. **Source == linkedin_post**
     LinkedIn posts carry richer recruiter intent than job-board listings
     (direct hiring-manager signal, curated comments, often a real email to
     apply). The user explicitly asked for this source to rank above others.

  2. **Apply URL appears in comments**
     A post where the recruiter DROPPED the apply link in comments is almost
     always a real, actively-recruiting hire (vs. a post that just has the
     link buried in auto-fill). Strongest intent signal we have.

  3. **Hiring company in Indian tier-1 list**
     The user prefers well-known Indian tech unicorns / startups. We load
     the list from config/indian_tier1_companies.json at module import;
     match is case-insensitive against structured fields (company,
     company_name, author_name) and — as a last resort — against the first
     ~600 chars of JD body (where the hiring company is usually named).

  4. **Remote**
     Detected via structured location field or JD regex ("fully remote",
     "WFH", "anywhere in india", "remote-first", etc.).

  5. **Location matches a user city**
     Small boost for Gurgaon / Delhi-NCR / Bangalore / Mumbai etc.

All boost magnitudes live in config.py (PRIORITY_BOOST_*) so they're
tunable without touching this module.
"""
from __future__ import annotations

import csv
import json
import re
from typing import Any

from . import config

# ----------------------------- Remote detection -----------------------------
_REMOTE_PATTERNS = [
    re.compile(r"\bfully\s+remote\b", re.I),
    re.compile(r"\b100\s*%\s*remote\b", re.I),
    re.compile(r"\bremote[\s\-]?first\b", re.I),
    re.compile(r"\b(?:work\s+from\s+home|wfh|work\s*@\s*home)\b", re.I),
    re.compile(r"\banywhere\s+in\s+india\b", re.I),
    re.compile(r"\bremote\s+(?:job|role|position|opportunity|opening)\b", re.I),
    re.compile(r"\blocation\s*[:\-–]\s*remote\b", re.I),
    re.compile(r"\bremote\s*[,/]\s*india\b", re.I),
]

# ----------------------------- Tier-1 company list -----------------------------
_TIER1_COMPANIES_CACHE: set[str] | None = None


def _load_tier1_companies() -> set[str]:
    path = config.INDIAN_TIER1_COMPANIES_PATH
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return set()
    companies = data.get("companies") if isinstance(data, dict) else data
    if not isinstance(companies, list):
        return set()
    return {
        c.strip().lower()
        for c in companies
        if isinstance(c, str) and c.strip() and not c.startswith("_")
    }


def _tier1() -> set[str]:
    global _TIER1_COMPANIES_CACHE
    if _TIER1_COMPANIES_CACHE is None:
        _TIER1_COMPANIES_CACHE = _load_tier1_companies()
    return _TIER1_COMPANIES_CACHE


def reload_tier1_companies() -> None:
    """Force a re-read of the tier-1 JSON. Used by tests that monkeypatch the
    config path mid-run; normal code paths rely on the module-load cache."""
    global _TIER1_COMPANIES_CACHE
    _TIER1_COMPANIES_CACHE = None


def _company_fields(post: dict[str, Any]) -> list[str]:
    """Structured company/author fields, in precedence order."""
    out: list[str] = []
    for key in ("company", "company_name", "author_name"):
        v = post.get(key)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


def _is_indian_tier1(post: dict[str, Any]) -> bool:
    """True if the hiring company matches a tier-1 Indian company.

    Precision > recall: we'd rather miss a boost than inflate a non-Indian
    role's score. So:
      - Structured fields are checked first (high precision, often exact name).
      - Substring match on structured fields (catches "Razorpay Software Pvt
        Ltd" → "razorpay").
      - JD-body scan is last-resort, restricted to first 600 chars AND with
        word-boundary regex so "Ex-Flipkart" doesn't trigger Flipkart.
    """
    companies = _tier1()
    if not companies:
        return False

    # 1. Structured fields, exact + substring match
    for raw in _company_fields(post):
        name = raw.lower()
        if name in companies:
            return True
        for c in companies:
            if len(c) >= 4 and c in name:
                return True

    # 2. JD body scan (head only), word-boundary to avoid casual mentions
    body = (post.get("jd_text") or post.get("text") or "").lower()
    if not body:
        return False
    head = body[:600]
    for c in companies:
        if len(c) < 5:
            # too short → high false-positive risk in free text
            continue
        if re.search(rf"(?<![\w.]){re.escape(c)}(?![\w.])", head):
            return True
    return False


def _is_remote(post: dict[str, Any]) -> bool:
    loc = (post.get("location") or "").lower()
    if loc and ("remote" in loc or "work from home" in loc or "wfh" in loc):
        return True
    body = post.get("jd_text") or post.get("text") or ""
    return any(p.search(body) for p in _REMOTE_PATTERNS)


def _is_user_city(post: dict[str, Any]) -> bool:
    loc = (post.get("location") or "").lower()
    if not loc:
        return False
    return any(city in loc for city in config.USER_CITIES)


# India geolocation — broad but precise. The INDIA_TOKENS list catches the
# explicit country name, every major city that recruiters actually name, and
# the most common state tokens. A bare mention like "Bengaluru" in a post
# body isn't evidence enough (the JD-extractor can inject city names as
# relocation options), so JD-body matching requires a location-declarative
# preamble ("based in …", "location: …") same as the role filter.
#
# Negatives list: tokens that indicate the role is DEFINITIVELY outside
# India. Fires the "not India" penalty. Kept conservative — we only match
# when the structured location field names one of these regions.
_INDIA_TOKENS = (
    "india", " in ",  # reserved, see below
    "bengaluru", "bangalore", "hyderabad", "mumbai", "pune", "delhi",
    "gurgaon", "gurugram", "noida", "chennai", "kolkata", "ahmedabad",
    "kochi", "kerala", "jaipur", "chandigarh", "indore", "trivandrum",
    "thiruvananthapuram", "coimbatore", "vizag", "visakhapatnam",
    "nagpur", "lucknow", "bhubaneswar", "mysore", "mysuru", "surat",
    "karnataka", "maharashtra", "tamil nadu", "telangana", "gujarat",
    "haryana", "uttar pradesh", "west bengal", "kerala", "punjab",
    "ncr", "delhi ncr", "delhi-ncr",
)
# Strip the reserved " in " — too noisy as a substring; we use it below
# only as part of phrase "in india".
_INDIA_TOKENS_SET = {t for t in _INDIA_TOKENS if t.strip() and t.strip() != "in"}

_NON_INDIA_LOCATION_TOKENS = (
    # US
    "united states", "usa", "u.s.a", "us ", ", us", " ca", ", california",
    "new york", "san francisco", "seattle", "austin", "boston", "chicago",
    "los angeles", "texas", "florida",
    # Europe
    "united kingdom", "uk ", ", uk", "london", "berlin", "amsterdam",
    "paris", "dublin", "munich", "zurich", "stockholm", "madrid",
    "barcelona", "copenhagen", "oslo",
    # SEA / APAC (excl. India)
    "singapore", "sydney", "melbourne", "tokyo", "seoul", "hong kong",
    "shanghai", "beijing", "bangkok", "jakarta", "manila", "kuala lumpur",
    # Middle East
    "dubai", "abu dhabi", "riyadh", "doha",
    # Canada / LatAm
    "toronto", "vancouver", "mexico city", "são paulo", "buenos aires",
)


def _is_india_located(post: dict[str, Any]) -> bool:
    """True if the post's location field OR JD-body location preamble names
    an Indian city/state/country. Kept precise: "India" or one of ~25 city/
    state tokens in the structured field is a lock; body-level matches
    require a "based in", "location:", or "office in" preamble to avoid
    flagging any JD that casually mentions Bengaluru."""
    loc = (post.get("location") or "").lower()
    if loc:
        if "india" in loc:
            return True
        if any(tok in loc for tok in _INDIA_TOKENS_SET):
            return True
    # JD-body fallback: recruiter posts often skip the structured location.
    # STRICT: only fire on a location-declarative preamble ("based in …",
    # "location: …", "office in …"). Bare "in India" mentions are too common
    # in JDs that name India only as a market ("customers in India, US, EU").
    body = (post.get("jd_text") or post.get("text") or "").lower()
    if not body:
        return False
    head = body[:800]
    preamble = r"(?:based\s+in|location\s*[:\-]|office\s+in|headquartered\s+in|hq\s+in|posted\s+in)\s+"
    for tok in _INDIA_TOKENS_SET:
        if len(tok) < 4:
            continue
        if re.search(preamble + r"[\w\s,\-]{0,40}?" + re.escape(tok), head):
            return True
    # "pan-india" / "pan india" is unambiguous recruiter-speak for a role
    # covering the whole country. Not matched as a casual mention elsewhere.
    if re.search(r"\bpan[\s\-]?india\b", head):
        return True
    return False


def _is_non_india_located(post: dict[str, Any]) -> bool:
    """Conservative counterpart: structured location names a non-India region
    AND no India signal fires. Used to push down US/EU/SEA roles that leaked
    through via lexically similar JDs."""
    loc = (post.get("location") or "").lower()
    if not loc:
        return False  # no structured location → stay neutral
    if any(tok in loc for tok in _NON_INDIA_LOCATION_TOKENS):
        return not _is_india_located(post)
    return False


def _walk_comments(comments: Any) -> list[dict[str, Any]]:
    """Flatten every comment + nested reply into a single flat list, preserving
    author/text fields so we can distinguish OP-authored replies downstream."""
    out: list[dict[str, Any]] = []
    if not isinstance(comments, list):
        return out
    for c in comments:
        if not isinstance(c, dict):
            continue
        out.append(c)
        replies = c.get("replies")
        if isinstance(replies, list):
            out.extend(_walk_comments(replies))
    return out


def _walk_comment_texts(comments: Any) -> list[str]:
    return [c.get("text") or "" for c in _walk_comments(comments) if c.get("text")]


def _apply_link_in_comments(post: dict[str, Any]) -> tuple[bool, bool]:
    """Return (in_any_comment, from_op_reply).

    - in_any_comment: True iff an apply URL/email appears in ANY comment or reply.
    - from_op_reply : True iff that match sits in a comment authored by the POST
                      AUTHOR themselves — the textbook recruiter pattern where a
                      hiring manager says "I'll drop the link in comments" and
                      then self-replies with the form. Strongest intent we can
                      detect without a paid LLM pass on the comment tree.

    Job-board sources don't have comments — we skip them entirely.
    """
    if post.get("source") != "linkedin_post":
        return False, False
    apply_urls = [u for u in (post.get("apply_urls") or []) if u]
    apply_emails = [e for e in (post.get("apply_emails") or []) if e]
    if not apply_urls and not apply_emails:
        return False, False

    post_author = (post.get("author_name") or "").strip().lower()
    comments = _walk_comments(post.get("comments"))
    if not comments:
        return False, False

    in_any = False
    from_op = False
    for c in comments:
        text = c.get("text") or ""
        if not text:
            continue
        author = c.get("author")
        # `author` can be a string (normalized shape), a dict (raw HarvestAPI
        # shape with a `.name`), or None (missing).
        if isinstance(author, dict):
            author_name = (author.get("name") or "").strip().lower()
        else:
            author_name = (author or "").strip().lower()

        hit = False
        for u in apply_urls:
            if u in text:
                hit = True
                break
        if not hit:
            lower_text = text.lower()
            for e in apply_emails:
                if e in lower_text:
                    hit = True
                    break
        if not hit:
            continue
        in_any = True
        if post_author and author_name and author_name == post_author:
            from_op = True
            break  # strongest signal; no need to scan further

    return in_any, from_op


# ----------------------------- Hiring-intent author -----------------------------
#
# A post by someone whose title screams "I hire people" is a much stronger signal
# than a reshare by a random SWE. These are the patterns the user explicitly
# called out as the target archetype ("hiring manager, recruiters, HRs or
# company folks"). Match is case-insensitive + word-boundary against author_title.
#
# We also accept founder/CEO/CTO/CHRO because in Indian startups those folks
# frequently post hiring directly — they ARE the hiring channel for early hires.
_HIRING_AUTHOR_PAT = re.compile(
    r"\b(?:"
    r"recruiter|recruitment|recruiting|"
    r"talent\s+(?:acquisition|partner|advisor|specialist|sourcer|lead|head|manager)|"
    r"\bta\b(?:\s+partner|\s+specialist|\s+lead|\s+manager)?|"
    r"hr\s+(?:business\s+partner|manager|lead|head|director|associate|generalist)|"
    r"\bhrbp\b|"
    r"hiring\s+(?:manager|lead)|"
    r"head\s+of\s+(?:talent|people|hr|recruiting|recruitment|hiring)|"
    r"people\s+(?:ops|operations|partner|lead|team)|"
    r"chief\s+people\s+officer|chro|chpo|"
    r"founder|co[\-\s]?founder|"
    r"chief\s+executive|\bceo\b|"
    r"chief\s+technology|\bcto\b|"
    r"chief\s+operating|\bcoo\b|"
    r"chief\s+product|\bcpo\b|"
    r"chief\s+marketing|\bcmo\b|"
    r"chief\s+of\s+staff"
    r")\b",
    re.I,
)


def is_hiring_intent_author(post: dict[str, Any]) -> bool:
    """True when the post author's title matches a hirer archetype.

    Only meaningful for linkedin_post (job-board sources don't carry an
    author_title). We guard accordingly so a Naukri listing with an empty
    title doesn't accidentally match an empty regex.

    Public API: also used by `application_filter` to gate the soft-apply
    (DM-me / comment-interested) channel so noisy non-hiring posts with
    "DM me" language don't flood the funnel.
    """
    if post.get("source") != "linkedin_post":
        return False
    title = (post.get("author_title") or "").strip()
    if not title:
        return False
    return bool(_HIRING_AUTHOR_PAT.search(title))


# Back-compat alias for code that imported the private name before the
# extraction to a public API (was in-module until the application_filter
# soft-apply gate needed to share the signal).
_is_hiring_intent_author = is_hiring_intent_author


# ----------------------------- Recruiter-style language -----------------------------
#
# Even when the post AUTHOR's title isn't in the recruiter archetype list
# (_HIRING_AUTHOR_PAT above), the post BODY may still carry the unmistakable
# fingerprint of a hiring post authored by someone inside the hiring company.
# Examples from the target archetype the user called out:
#
#   • "We're hiring 25 people at Cars24 to build …"
#   • "Trupeer's hiring Content Marketer, Product Manager …"
#   • "At Scaler we are looking for a Strategy …"
#   • "Looking for an Associate Chief of Staff who …"
#   • "Looking to connect with professionals with …"
#
# These posts are typically written by a team lead / growth person / IC who
# ISN'T a "Recruiter" by title but IS actively hiring for their own team.
# The language is disproportionately specific to hiring posts — general
# business/networking content doesn't say "we're hiring 3 PMs".
#
# Design notes on precision (avoid false positives):
#   • "hiring trends in 2026" — a thinkpiece. We require the word `hiring`
#     to be adjacent to a first-person-plural / agent ("we are / we're /
#     now / currently / actively / <Company>'s / <Company> is").
#   • "Just got hired at X!" — a celebration post. Passive-voice "hired"
#     is excluded; we match active "hiring" only.
#   • "Looking for feedback" — an info-request post. We require "looking
#     for a/an <noun>" with an indefinite article OR "looking to
#     hire/connect/recruit/onboard".
#   • Hashtag-only posts: `#hiring` alone is ambiguous (could be a reshare).
#     We count hashtags as CORROBORATING evidence, not a primary trigger.
#
# Stacking: this boost is ADDITIVE with _is_hiring_intent_author. If both
# fire — recruiter title + recruiter language — we're basically certain.
# The boost size stays modest (~0.5) so stacking doesn't runaway; two
# signals on a well-scored post nudge it past single-signal peers.
_RECRUITER_PHRASE_PATTERNS = [
    # "we're hiring" / "we are hiring" / "we're recruiting" / "we are recruiting"
    re.compile(r"\bwe(?:'re|'?re|\s+are)\s+(?:hiring|recruiting|looking\s+to\s+hire|growing|expanding\s+the\s+team)\b", re.I),
    # "<Company>'s hiring" / "<Company> is hiring" — capitalized token anchor
    # (avoid "it's hiring", "he's hiring" by requiring at least 2 word chars
    # before the possessive)
    re.compile(r"\b[A-Z][\w&.\-]{2,}(?:'s|'?s|\s+is)\s+hiring\b"),
    # "now hiring" / "currently hiring" / "actively hiring"
    re.compile(r"\b(?:now|currently|actively|urgently|immediately)\s+hiring\b", re.I),
    # "hiring <N>? <role words>" — e.g. "hiring 25 people", "hiring 3 Product Managers"
    # Requires a role-ish word to follow so "hiring market" trend posts miss.
    re.compile(r"\bhiring\s+(?:\d+\s+)?(?:people|candidates|folks|engineers?|developers?|designers?|managers?|analysts?|associates?|leads?|heads?|directors?|consultants?|interns?|specialists?|marketers?|recruiters?|architects?|writers?|founders?|executives?|professionals?|operators?|scientists?|researchers?|strategists?|officers?|talents?|teammates?|colleagues?)\b", re.I),
    # "looking for a/an <role>" — require indefinite article + word char so
    # "looking for feedback" misses but "looking for a PM" matches.
    re.compile(r"\blooking\s+for\s+(?:an?|our\s+next|the\s+next|a\s+great|a\s+new)\s+\w", re.I),
    # "looking to hire / recruit / onboard" — active hiring verbs
    re.compile(r"\blooking\s+to\s+(?:hire|recruit|onboard|bring\s+(?:on|in)|fill)\b", re.I),
    # "looking to connect with <professionals/candidates/...>" — recruiter outreach
    re.compile(r"\blooking\s+to\s+connect\s+with\s+(?:professionals|candidates|people|folks|experienced|senior|junior|passionate|talent(?:ed)?)", re.I),
    # "at <Company> we are looking/hiring/recruiting" — strongest specific pattern
    re.compile(r"\bat\s+[A-Z][\w&.\-]{1,}(?:\s+[A-Z][\w&.\-]{1,})?\s+we(?:'re|'?re|\s+are)\s+(?:looking|hiring|recruiting|growing|expanding|scaling)\b"),
    # "join us at <Company>" / "join our team" / "join my team"
    re.compile(r"\bjoin\s+(?:us\s+at\s+|our\s+|my\s+)(?:team|company|mission|journey|growing\s+team)\b", re.I),
    # "<N> positions/roles/openings/vacancies at/in/for"
    re.compile(r"\b(?:\d+\s+)?(?:open\s+)?(?:position|role|opening|opportunity|vacanc|spot)(?:s|ies)?\s+(?:are\s+)?(?:open\s+)?(?:at|in|for|with)\b", re.I),
    # "we have <N> <role> openings / positions"
    re.compile(r"\bwe\s+have\s+(?:\d+|a|an|several|multiple|many|some|new)\s+(?:\w+\s+){0,3}(?:opening|position|role|vacanc|spot)", re.I),
    # "apply now" / "apply here" / "apply using <link>" — recruiter CTA
    re.compile(r"\bapply\s+(?:now|here|today|using|via|by|through|at|through\s+the\s+link|via\s+the\s+link)\b", re.I),
    # "drop your resume / CV / portfolio" — active ask
    re.compile(r"\bdrop\s+(?:your\s+|a\s+)?(?:resume|cv|portfolio|profile|application)\b", re.I),
    # "we need <an? word>" — urgent hiring
    re.compile(r"\bwe\s+(?:need|want)\s+(?:an?|more|another)\s+\w", re.I),
    # "excited to (be hiring / announce / share) … hiring" — celebratory hire announcement
    re.compile(r"\bexcited\s+to\s+(?:be\s+hiring|announce|share)\b[^.!?\n]*\bhiring\b", re.I),
]

# Weighted hashtag scoring.
#
# Historical rule was "≥2 hashtags to fire", but that dropped single
# high-signal hashtags like `#nowhiring` (86K followers, overwhelmingly
# used by actual recruiters) while keeping any two generic tags
# (`#jobs #careers` — often reshares). Each hashtag now carries a weight
# proportional to log(LinkedIn follower count), and the signal fires when
# the sum of matched weights exceeds a single-strong-tag threshold.
#
# Weights were calibrated from public LinkedIn follower counts (rounded)
# so `#nowhiring` (3.0) fires alone, `#hiring` (2.4) sits just below the
# threshold and needs corroboration, and junk like `#joboftheday` or
# `#jobalert` requires multiple companions.
#
# Threshold 2.5 → matches historical "one generic #hiring wasn't enough"
# behavior. `#nowhiring`, `#hiringalert`, or two mid-weight tags clear it.
_RECRUITER_HASHTAG_WEIGHTS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"#nowhiring\b", re.I),        3.0),
    (re.compile(r"#hiringnow\b", re.I),        3.0),
    (re.compile(r"#hiringalert\b", re.I),      2.8),
    (re.compile(r"#urgenthiring\b", re.I),     2.8),
    (re.compile(r"#wearehiring\b", re.I),      2.6),
    (re.compile(r"#hiring\b", re.I),           2.4),
    (re.compile(r"#recruiting\b", re.I),       2.0),
    (re.compile(r"#hiringmanagers?\b", re.I),  1.8),
    (re.compile(r"#openposition\b", re.I),     1.5),
    (re.compile(r"#openroles?\b", re.I),       1.5),
    (re.compile(r"#jobopening\b", re.I),       1.4),
    (re.compile(r"#jobopportunity\b", re.I),   1.3),
    (re.compile(r"#careeropportunity\b", re.I),1.3),
    (re.compile(r"#joinus\b", re.I),           1.2),
    (re.compile(r"#jobalert\b", re.I),         1.0),
    (re.compile(r"#joboftheday\b", re.I),      0.9),
    (re.compile(r"#jobs\b", re.I),             0.6),
    (re.compile(r"#careers\b", re.I),          0.6),
]

_RECRUITER_HASHTAG_THRESHOLD = 2.5


def has_recruiter_language(post: dict[str, Any]) -> bool:
    """True when the POST BODY reads like a hiring-manager / team-lead post.

    Two independent triggers (either fires the signal):
      1. ≥1 strong phrase pattern matches the body text (the common case).
      2. Weighted hashtag sum ≥ threshold (covers the hashtag-cluster
         archetype: `#nowhiring` alone fires; `#hiring + #joinus` fires;
         a single `#careers` does not).

    Source gate: LinkedIn posts only. Job-board listings (Naukri, LinkedIn
    Jobs) have machine-generated JD text where these patterns would misfire.

    Public API: also used by `application_filter` to gate the soft-apply
    (DM-me / comment-interested) channel.
    """
    if post.get("source") != "linkedin_post":
        return False
    body = post.get("text") or ""
    if not body:
        return False

    # Primary trigger: a strong phrase
    if any(p.search(body) for p in _RECRUITER_PHRASE_PATTERNS):
        return True

    # Secondary trigger: weighted hashtag sum — one strong tag or a
    # cluster of medium ones clears the bar.
    weight_sum = sum(w for p, w in _RECRUITER_HASHTAG_WEIGHTS if p.search(body))
    return weight_sum >= _RECRUITER_HASHTAG_THRESHOLD


# Back-compat alias.
_has_recruiter_language = has_recruiter_language

# Legacy pattern list kept for any downstream code that inspected it.
# Regenerated from the weighted list so it stays in sync.
_RECRUITER_HASHTAG_PATTERNS = [p for p, _ in _RECRUITER_HASHTAG_WEIGHTS]


# ----------------------------- Connection degree -----------------------------
#
# HarvestAPI cannot expose connection degree to the querying account (it
# scrapes via its own LinkedIn accounts, not the user's). Workaround:
# the user exports their LinkedIn Connections CSV from Settings → Data
# Privacy → Get a copy of your data → Connections. We match the post's
# author_name against that export to detect 1st-degree.
#
# 2nd-degree is inferred heuristically: when the post author's current
# company appears in the user's 1st-degree connections' companies, the
# user almost certainly has a shared connection at that company — so the
# post author is very likely 2nd-degree via that path. This is a precision
# trade-off (some false positives on companies with thousands of
# employees), offset by the smaller boost magnitude.
#
# Caches are loaded lazily and can be reset in tests via
# `reload_connections()`. If the CSV is missing both sets are empty and
# all connection checks no-op — the feature is opt-in.

_CONNECTIONS_NAMES_CACHE: set[str] | None = None
_CONNECTIONS_COMPANIES_CACHE: set[str] | None = None

# Words that attach to company names and obscure the canonical token.
# Stripping is done in a LOOP because several of these chain (e.g.
# "Razorpay Software Pvt Ltd" — three passes to reduce to "razorpay"):
#   Pass 1: strip "ltd"       → "razorpay software pvt"
#   Pass 2: strip "pvt"       → "razorpay software"
#   Pass 3: strip "software"  → "razorpay"
# List is kept conservative — we do NOT strip words that could be part
# of a genuinely-distinct company name ("Razorpay Labs" should keep the
# "labs" if the user's connection is specifically at Razorpay Labs).
_COMPANY_SUFFIX_RE = re.compile(
    r"\s+(?:pvt\.?|private|ltd\.?|limited|inc\.?|llc|llp|corporation|corp\.?|"
    r"company|co\.?|technologies|tech|solutions|services|industries|group|"
    r"holdings|software|india)\s*$",
    re.I,
)


def _strip_company_suffixes(raw: str) -> str:
    """Loop-strip legal / corporate suffixes. Shared by both the connections
    loader and the 2nd-degree matcher so both sides of a comparison are
    normalized the same way."""
    if not raw:
        return ""
    s = raw.strip().lower()
    prev: str | None = None
    while prev != s:
        prev = s
        s = _COMPANY_SUFFIX_RE.sub("", s).strip()
    return s


def _load_connections() -> tuple[set[str], set[str]]:
    """Parse the LinkedIn connections CSV into (names, companies) sets.

    LinkedIn's native export prepends a few "Notes:" lines above the real
    header. We skip any line that isn't a valid CSV header, then use
    DictReader for the rest. Names are stored lowercase+whitespace-
    collapsed; companies too, with a length filter so single-letter noise
    doesn't leak in and legal suffixes loop-stripped (see regex above).
    """
    path = config.LINKEDIN_CONNECTIONS_PATH
    if not path.exists():
        return set(), set()

    names: set[str] = set()
    companies: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            # LinkedIn's export has "Notes:" + "When ..." preamble lines
            # before the real header. Walk until we find a row that looks
            # like the header, then rewind via an in-memory buffer.
            lines = f.readlines()
    except OSError:
        return set(), set()

    header_idx = next(
        (
            i for i, line in enumerate(lines)
            if line.lower().lstrip().startswith("first name")
            or line.lower().lstrip().startswith('"first name"')
        ),
        None,
    )
    if header_idx is None:
        return set(), set()

    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        first = (row.get("First Name") or "").strip()
        last = (row.get("Last Name") or "").strip()
        full = re.sub(r"\s+", " ", f"{first} {last}").strip().lower()
        if full:
            names.add(full)
        company = _strip_company_suffixes(row.get("Company") or "")
        if len(company) >= 3:
            companies.add(company)

    return names, companies


def _connections() -> tuple[set[str], set[str]]:
    global _CONNECTIONS_NAMES_CACHE, _CONNECTIONS_COMPANIES_CACHE
    if _CONNECTIONS_NAMES_CACHE is None or _CONNECTIONS_COMPANIES_CACHE is None:
        names, companies = _load_connections()
        _CONNECTIONS_NAMES_CACHE = names
        _CONNECTIONS_COMPANIES_CACHE = companies
    return _CONNECTIONS_NAMES_CACHE, _CONNECTIONS_COMPANIES_CACHE


def reload_connections() -> None:
    """Force re-read of the connections CSV. Used by tests and by any
    ops workflow that rewrites the file mid-run."""
    global _CONNECTIONS_NAMES_CACHE, _CONNECTIONS_COMPANIES_CACHE
    _CONNECTIONS_NAMES_CACHE = None
    _CONNECTIONS_COMPANIES_CACHE = None


# Common join words that appear in headlines and should not be matched
# as "at <Company>" (e.g. "at scale" is a phrase, not a company).
_AT_STOPWORDS = {"scale", "home", "work", "once", "heart", "best", "worst"}


def _extract_author_company(post: dict[str, Any]) -> str | None:
    """Best-effort extraction of the post author's current company from
    the HarvestAPI `author_title` / headline string.

    Supported shapes (in order of preference):
      - "Head of Growth at Razorpay"            → "razorpay"
      - "Head of Growth | Razorpay"             → "razorpay"
      - "Head of Growth - Razorpay"             → "razorpay"
      - "Head of Growth @ Razorpay"             → "razorpay"

    Returns None when no confident match — the 2nd-degree signal only
    fires when we can pinpoint a single company.
    """
    title = (post.get("author_title") or "").strip()
    if not title:
        # Structured `company` field — some sources populate it directly.
        company = (post.get("company") or "").strip()
        return company.lower() or None if company else None

    # "X at Y" — the dominant LinkedIn headline pattern
    m = re.search(
        r"\s+at\s+([A-Za-z0-9&.,'\-\s]{2,60}?)(?:\s*[\|,\-•–—·]|\s*$)",
        title,
        re.I,
    )
    if m:
        company = m.group(1).strip().lower()
        if company and company not in _AT_STOPWORDS:
            return company

    # "X @ Y"
    m = re.search(r"\s+@\s+([A-Za-z0-9&.,'\-\s]{2,60}?)(?:\s*[\|,]|\s*$)", title)
    if m:
        return m.group(1).strip().lower() or None

    # "Role | Company" / "Role, Company" / "Role - Company"
    parts = re.split(r"\s*[\|,\-•–—]\s*", title)
    if len(parts) >= 2:
        last = parts[-1].strip().lower()
        # Skip if the last token looks like a description, not a company.
        if 3 <= len(last) <= 60 and not re.search(r"\b(i|we|my|the|and|or)\b", last):
            return last

    return None


def _is_first_degree_connection(post: dict[str, Any]) -> bool:
    """True iff the post author's name matches a name in the user's
    LinkedIn 1st-degree connections export. LinkedIn posts only — job
    boards don't carry a real author_name."""
    if post.get("source") != "linkedin_post":
        return False
    names, _ = _connections()
    if not names:
        return False
    author = (post.get("author_name") or "").strip().lower()
    if not author:
        return False
    author = re.sub(r"\s+", " ", author)
    return author in names


def _is_second_degree_connection(post: dict[str, Any]) -> bool:
    """True iff the post author works at a company where the user has at
    least one 1st-degree connection. Strong proxy for 2nd-degree since
    a shared connection at the same company is almost always mutual.

    Precision guard: we skip the check when the author is already a
    1st-degree connection (they're 1st, not 2nd; caller awards only one
    of the two boosts to avoid double-counting).

    Matching is symmetric — we normalize both the parsed author company
    AND each connection company with the same suffix-stripper, then do
    exact / whole-word substring match in either direction. That handles
    the "Razorpay" vs "Razorpay Software Pvt Ltd" mismatch without
    misfiring on unrelated prefix collisions (`\\b` word boundaries
    reject "Meta" ↔ "Metal Tech").
    """
    if post.get("source") != "linkedin_post":
        return False
    _, companies = _connections()
    if not companies:
        return False
    author_raw = _extract_author_company(post)
    if not author_raw:
        return False
    author_company = _strip_company_suffixes(author_raw)
    if not author_company:
        return False
    if author_company in companies:
        return True
    # Whole-word substring in either direction — prevents "meta" in "metal".
    for c in companies:
        if len(c) >= 4 and re.search(rf"\b{re.escape(c)}\b", author_company):
            return True
        if len(author_company) >= 4 and re.search(rf"\b{re.escape(author_company)}\b", c):
            return True
    return False


def compute_priority_boost(post: dict[str, Any]) -> tuple[float, list[str]]:
    """Return (total_boost, reason_labels).

    Labels are short human strings used for email badges. Multiple labels
    are possible (e.g. a LinkedIn post from a Tier-1 Indian company that's
    remote-friendly would collect three badges).

    Signal stacking is intentional — a LinkedIn post where the recruiter
    themselves dropped the form in comments AND is tagged "Head of Talent"
    at a Tier-1 Indian company should visibly dominate the ranking.
    """
    boost = 0.0
    reasons: list[str] = []

    if post.get("source") == "linkedin_post":
        boost += config.PRIORITY_BOOST_LINKEDIN_POST
        reasons.append("LinkedIn post")

    # Apply-link-in-comments: also check whether the comment came from the
    # post author (strongest recruiter pattern). We add BOTH boosts so an
    # OP-curated drop strictly dominates a random-commenter paste.
    in_any_comment, from_op = _apply_link_in_comments(post)
    if in_any_comment:
        boost += config.PRIORITY_BOOST_APPLY_IN_COMMENTS
        reasons.append("Apply in comments")
    if from_op:
        boost += config.PRIORITY_BOOST_APPLY_FROM_OP
        reasons.append("Recruiter-dropped link")

    if is_hiring_intent_author(post):
        boost += config.PRIORITY_BOOST_HIRING_INTENT
        reasons.append("Hiring manager / recruiter")

    # Body-language signal: fires when the POST TEXT itself reads like a
    # recruiter's / team-lead's hiring post (vs. a reshare or a generic
    # industry thinkpiece). Additive with the author-title signal — both
    # matching = high-confidence recruiter drop.
    if has_recruiter_language(post):
        boost += config.PRIORITY_BOOST_RECRUITER_LANGUAGE
        reasons.append("Recruiter-style post")

    # Connection-degree signals. 1st-degree short-circuits 2nd (no
    # double-counting); both are LinkedIn-posts-only via inner guards.
    is_1st = _is_first_degree_connection(post)
    if is_1st:
        boost += config.PRIORITY_BOOST_CONNECTION_1ST
        reasons.append("1st connection")
    elif _is_second_degree_connection(post):
        boost += config.PRIORITY_BOOST_CONNECTION_2ND
        reasons.append("2nd connection")

    indian_tier1 = _is_indian_tier1(post)
    if indian_tier1:
        boost += config.PRIORITY_BOOST_INDIAN_TIER1
        reasons.append("Tier-1 Indian company")

    if _is_remote(post):
        boost += config.PRIORITY_BOOST_REMOTE
        reasons.append("Remote")

    user_city = _is_user_city(post)
    if user_city:
        boost += config.PRIORITY_BOOST_USER_CITY
        reasons.append("Near you")

    # India-location: only fire when neither of the two stronger Indian
    # signals already did — otherwise we'd stack +1.0 (tier1) + +0.3 (city)
    # + +0.4 (india) on the same physical cue.
    if not indian_tier1 and not user_city and _is_india_located(post):
        boost += config.PRIORITY_BOOST_INDIA
        reasons.append("India")
    # Non-India penalty: structured location names a non-India region and
    # no India signal fires. Subtract a small amount so Indian roles
    # dominate even when the fit score is comparable. Guarded by the
    # configured magnitude so setting it to 0 disables the penalty cleanly.
    elif config.PRIORITY_PENALTY_NON_INDIA > 0 and _is_non_india_located(post):
        boost -= config.PRIORITY_PENALTY_NON_INDIA
        reasons.append("Not in India")

    return boost, reasons


def apply_priority_boosts(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each post, compute the boost, add to llm_score (clamped to 10),
    store the base/boost/reasons for transparency, then re-sort desc.

    Adds these keys to each post:
      - base_llm_score (float)    — pre-boost Haiku score
      - priority_boost (float)    — total boost applied
      - priority_reasons (list)   — badge labels for email rendering
      - llm_score (float)         — post-boost score (clamped to 10)
    """
    out: list[dict[str, Any]] = []
    for p in posts:
        boost, reasons = compute_priority_boost(p)
        base = float(p.get("llm_score", 0.0))
        new_score = max(0.0, min(10.0, base + boost))
        out.append(
            {
                **p,
                "base_llm_score": base,
                "priority_boost": round(boost, 3),
                "priority_reasons": reasons,
                "llm_score": round(new_score, 3),
            }
        )
    out.sort(key=lambda p: p["llm_score"], reverse=True)
    return out
