"""Central configuration."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
RESUME_DIR = Path(os.getenv("RESUME_DIR", ROOT / "resume"))
CONFIG_DIR = ROOT / "config"
RESULTS_DIR = ROOT / "results"

PREFS_PATH = CONFIG_DIR / "role_preferences.json"
PROFILE_PATH_TMPL = CONFIG_DIR / "resume_profile_{track}.json"

# Secrets
APIFY_TOKEN = os.getenv("APIFY_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")

# Email delivery: "resend" (recommended, no password) or "gmail" (SMTP + app password)
DELIVERY_METHOD = os.getenv("DELIVERY_METHOD", "resend").lower()
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
# Resend requires a verified sender domain; their shared onboarding@resend.dev
# sender works for personal use without domain verification.
RESEND_FROM = os.getenv("RESEND_FROM", "LinkedIn Scraper <onboarding@resend.dev>")
EMAIL_TO = os.getenv("EMAIL_TO") or GMAIL_ADDRESS

# LinkedIn search config
CITIES = ["Bangalore", "Bengaluru", "Mumbai", "Pune", "Delhi", "Gurgaon", "Noida"]
INCLUDE_REMOTE = True
POSTED_LIMIT_DAYS = int(os.getenv("POSTED_LIMIT_DAYS", "7"))
MAX_POSTS_PER_QUERY = int(os.getenv("MAX_POSTS_PER_QUERY", "200"))
# Widened from 7 → 10 because the query builder no longer multiplies roles × cities;
# the ladder is roles × (T1..T5), so more slots = more tiers land before the cap.
QUERIES_PER_TRACK = int(os.getenv("QUERIES_PER_TRACK", "10"))

# ---------- Multi-source scraping knobs ----------
# Ordered comma-separated list of source modules to run. Each maps to
# src/sources/<name>.py. Default turns on all three; drop any you don't
# want paid hits on by editing ENABLED_SOURCES in .env or the workflow.
#
#   linkedin_posts  — harvestapi, $0.002/post (paid, premium signal)
#   linkedin_jobs   — worldunboxer/rapid-linkedin-scraper, FREE
#   naukri_jobs     — shahidirfan/Naukri-Jobs-Scraper, FREE
ENABLED_SOURCES = [
    s.strip()
    for s in os.getenv(
        "ENABLED_SOURCES", "linkedin_posts,linkedin_jobs,naukri_jobs"
    ).split(",")
    if s.strip()
]

# Per-track cap for each job-board source. These default to 500 which —
# combined with LinkedIn posts at ~200 / track — lands us at roughly
# 3000-5000 raw items per run across 4 tracks, well inside the free-tier
# budget. Raise to widen the funnel, lower if Apify throttles you.
MAX_LINKEDIN_JOBS_PER_TRACK = int(os.getenv("MAX_LINKEDIN_JOBS_PER_TRACK", "500"))
MAX_NAUKRI_JOBS_PER_TRACK = int(os.getenv("MAX_NAUKRI_JOBS_PER_TRACK", "500"))
# How many role titles from each track's roles[] list to pass to the
# job-board actors. Top K by preference order — anything beyond that
# is long-tail and the return-per-call drops sharply.
ROLES_PER_JOB_BOARD_TRACK = int(os.getenv("ROLES_PER_JOB_BOARD_TRACK", "4"))

# Soft deadline for the scrape stage. When a workflow runs under a hard
# wall-clock ceiling (e.g. GitHub Actions `timeout-minutes: 60`), hitting
# that ceiling mid-scrape kills the whole process and throws away every
# post collected so far. This knob lets the orchestrator stop starting
# new tracks once N seconds have elapsed and return the partial dict to
# the pipeline — filter/match/rerank/deliver then runs normally on the
# tracks that did complete, so partial runs still produce an email.
#
# Default 2400s (40min) leaves ~20min of headroom inside a 60-min workflow
# for filter + JD extract + embed + rerank + deliver. Raise this if your
# workflow cap is larger; lower if you see the cap still biting.
SCRAPE_SOFT_DEADLINE_SECONDS = int(os.getenv("SCRAPE_SOFT_DEADLINE_SECONDS", "2400"))

# Matching thresholds
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.62"))
LLM_RERANK_TOP_N = int(os.getenv("LLM_RERANK_TOP_N", "30"))
# Reserve slots for LinkedIn posts in the rerank top-N so they can't be
# crowded out by machine-generated job-board JDs (Naukri, LinkedIn Jobs)
# that match the resume summary's JD-speak lexically but carry less
# recruiter intent. 15 of the 30 rerank slots are guaranteed to LinkedIn
# posts if that many clear the similarity threshold; the other 15 go to
# the highest-similarity posts from any source.
#
# Without this, the top-30 was saturated with Naukri + LinkedIn Easy
# Apply listings every run and the user's explicit priority — LinkedIn
# posts with recruiter-style hiring language — never reached the LLM.
LLM_RERANK_MIN_LINKEDIN_POSTS = int(os.getenv("LLM_RERANK_MIN_LINKEDIN_POSTS", "15"))
# Lowered from 7.0 → 6.5: with a 7-day window and broader apply-method catchment,
# we want borderline-fit posts surfaced rather than silently dropped.
LLM_SCORE_THRESHOLD = float(os.getenv("LLM_SCORE_THRESHOLD", "6.5"))

# ---------- Dedup / recency ----------
# Rolling window (in days) during which a delivered post is suppressed from
# future digests. Set to 0 to disable TTL (permanent dedup = original behavior).
# 3 days means: if a post was in today's email, it won't appear for the next
# 3 cron runs. On day 4 it becomes eligible again — useful for long-lived job
# board listings that would otherwise only ever be shown once.
DEDUP_DAYS = int(os.getenv("DEDUP_DAYS", "3"))

# ---------- Experience-level filter ----------
# The user's years-of-experience. JDs asking for significantly more YoE are
# hard-rejected BEFORE embedding/rerank (saves API spend AND fixes the "showing
# 10+ year senior roles" problem). Buffer allows some wiggle for optimistic
# "senior" framing — with USER_YOE=4.5 + YOE_BUFFER=3, a role asking for up to
# 7.5 years still passes. A role explicitly asking for 10+ years does not.
USER_YOE = float(os.getenv("USER_YOE", "4.5"))
YOE_BUFFER = float(os.getenv("YOE_BUFFER", "3.0"))

# ---------- Priority boosts (post-rerank) ----------
# Each signal adds its boost to the Haiku score. Final score is clamped to
# [0, 10]. Set a boost to 0 to disable that signal without deleting code.
#
#   LINKEDIN_POST       — source==linkedin_post (richer signal than job boards)
#   APPLY_IN_COMMENTS   — apply URL lives in post comments (curated recruiter
#                         drop vs. auto-generated job-board listing)
#   APPLY_FROM_OP       — the apply-in-comments reply came from the POST AUTHOR
#                         themselves (textbook recruiter pattern: "I'll drop the
#                         link in comments" → strongest hiring signal we have)
#   HIRING_INTENT       — author_title matches recruiter / HR / talent /
#                         founder / CEO / CTO / head-of keywords — i.e. the
#                         post is from an actual hirer, not a reshare
#   RECRUITER_LANGUAGE  — the POST BODY ITSELF reads like a recruiter / team
#                         lead post ("we're hiring 3 PMs at Leap", "Trupeer's
#                         hiring …", "at Scaler we are looking for …").
#                         Catches ICs posting about their own team's hire —
#                         orthogonal to HIRING_INTENT (which only inspects
#                         author_title). Additive: both firing = high
#                         confidence.
#   INDIAN_TIER1        — hiring company appears in config/indian_tier1_companies.json
#   REMOTE              — JD/location says fully-remote / WFH / anywhere-in-india
#   USER_CITY           — location matches a city in USER_CITIES (below)
PRIORITY_BOOST_LINKEDIN_POST = float(os.getenv("PRIORITY_BOOST_LINKEDIN_POST", "0.6"))
PRIORITY_BOOST_APPLY_IN_COMMENTS = float(os.getenv("PRIORITY_BOOST_APPLY_IN_COMMENTS", "0.4"))
PRIORITY_BOOST_APPLY_FROM_OP = float(os.getenv("PRIORITY_BOOST_APPLY_FROM_OP", "0.3"))
PRIORITY_BOOST_HIRING_INTENT = float(os.getenv("PRIORITY_BOOST_HIRING_INTENT", "0.4"))
PRIORITY_BOOST_RECRUITER_LANGUAGE = float(os.getenv("PRIORITY_BOOST_RECRUITER_LANGUAGE", "0.5"))
PRIORITY_BOOST_INDIAN_TIER1 = float(os.getenv("PRIORITY_BOOST_INDIAN_TIER1", "1.0"))
PRIORITY_BOOST_REMOTE = float(os.getenv("PRIORITY_BOOST_REMOTE", "0.5"))
PRIORITY_BOOST_USER_CITY = float(os.getenv("PRIORITY_BOOST_USER_CITY", "0.3"))
# India-location boost — fires when the post's structured location OR JD
# body geolocates the role in India (any Indian city / state / country
# token), and NONE of the stronger Indian signals (USER_CITY, TIER1) have
# already fired. Prevents non-India roles — US-based SaaS posts, APAC-wide
# roles with Singapore HQ, etc. — from crowding out India-based matches.
# Tuned conservatively (0.4) because Indian-tier1 already stacks a +1.0.
PRIORITY_BOOST_INDIA = float(os.getenv("PRIORITY_BOOST_INDIA", "0.4"))
# Symmetric negative boost for roles that we can confidently say are
# OUTSIDE India (US/EU/SEA/MENA/etc. with no India hook). Pushes them
# down so India roles dominate the ranking even when the fit score is
# comparable. Keep modest so a perfect-fit remote-global role isn't
# pathologically penalized — the user explicitly wants India prioritized,
# not everything-else banned.
PRIORITY_PENALTY_NON_INDIA = float(os.getenv("PRIORITY_PENALTY_NON_INDIA", "0.6"))

# ---------- Network-degree boosts ----------
# Fire when the post author is in the user's LinkedIn connections export.
# HarvestAPI cannot expose the querying account's network (it scrapes via its
# own LinkedIn accounts, not the user's), so we rely on the user's own export:
#   LinkedIn → Settings → Data Privacy → Get a copy of your data → Connections
# Drop the resulting Connections.csv at config/linkedin_connections.csv.
#
#   CONNECTION_1ST — direct author-name match against the export (exact).
#                    Strongest we can detect; warrants a meaningful boost.
#   CONNECTION_2ND — heuristic: author's current company appears in the user's
#                    1st-degree connections' companies. If the user has any
#                    connection at Acme Corp, a post by any Acme employee is
#                    very likely 2nd-degree via that shared connection. Not
#                    perfect but the only free-tier signal available.
PRIORITY_BOOST_CONNECTION_1ST = float(os.getenv("PRIORITY_BOOST_CONNECTION_1ST", "0.9"))
PRIORITY_BOOST_CONNECTION_2ND = float(os.getenv("PRIORITY_BOOST_CONNECTION_2ND", "0.45"))

# User-supplied LinkedIn connections CSV. Expected shape (from LinkedIn's
# native export): First Name, Last Name, URL, Email Address, Company,
# Position, Connected On. If the file doesn't exist, the connection signals
# silently no-op — keeps the feature opt-in.
LINKEDIN_CONNECTIONS_PATH = CONFIG_DIR / "linkedin_connections.csv"

# Curated list of well-known Indian tech companies / unicorns / soonicorns.
# Missing from the list = no boost (but post still ranked on Haiku score alone).
INDIAN_TIER1_COMPANIES_PATH = CONFIG_DIR / "indian_tier1_companies.json"

# ---------- Soft-apply gate ----------
# Many legit recruiter posts have no URL at all — they say "DM me your resume"
# or "comment 'interested' and I'll reach out". The default apply-method gate
# rejects these (no gform, no ATS URL, no email). Enabling soft-apply adds a
# new `soft_dm` method that fires on DM / comment-interested / drop-resume
# language, BUT only when the post ALSO carries hiring-intent signals (author
# headline matches recruiter archetype OR body reads like a recruiter post).
#
# Rationale: DM/comment language appears on non-hiring content too ("DM me
# your thoughts"), so ungated it would flood the funnel with noise. The
# two-gate structure keeps precision high.
ENABLE_SOFT_APPLY = os.getenv("ENABLE_SOFT_APPLY", "true").lower() == "true"

# ---------- In-digest dedup ----------
# After cross-track unique-assignment (pipeline stage 6) there can still be
# duplicates because the same role gets syndicated to LinkedIn posts +
# LinkedIn jobs + Naukri with different URLs. Content-hash dedup collapses
# them using (company, normalized_title, location) as a fuzzy key. Keeps
# the highest-scored instance.
ENABLE_CONTENT_DEDUP = os.getenv("ENABLE_CONTENT_DEDUP", "true").lower() == "true"

# Home cities for the user-city boost. Comma-separated, lowercased, matched
# as case-insensitive substring against the post's structured location field.
USER_CITIES = [
    c.strip().lower()
    for c in os.getenv(
        "USER_CITIES", "Gurgaon,Gurugram,Delhi,Noida,Bangalore,Bengaluru,Mumbai"
    ).split(",")
    if c.strip()
]

# Resume embedding cache — avoids a Voyage API call per track every run
# (the resume summary per track doesn't change between runs). Cache is
# keyed by SHA-256 of the summary text, so editing the resume silently
# invalidates old entries.
RESUME_EMBEDDING_CACHE = CONFIG_DIR / "resume_embedding_cache.json"

# Apify actor
APIFY_ACTOR_ID = "harvestapi/linkedin-post-search"

# Models
CLAUDE_MODEL_HEAVY = "claude-opus-4-7"
CLAUDE_MODEL_LIGHT = "claude-haiku-4-5-20251001"
VOYAGE_MODEL = "voyage-3.5"

# Google Form detection
GFORM_DIRECT_REGEXES = [
    r"https?://docs\.google\.com/forms/[^\s)>\]\"']+",
    r"https?://forms\.gle/[A-Za-z0-9_-]+",
    r"https?://forms\.google\.com/[^\s)>\]\"']+",
]
SHORTENER_HOSTS = (
    "lnkd.in",
    "bit.ly",
    "bitly.com",
    "tinyurl.com",
    "cutt.ly",
    "rebrand.ly",
    "rb.gy",
    "t.co",
    "t.ly",
    "shorturl.at",
    "is.gd",
    "ow.ly",
    "s.id",
    "linktr.ee",
    "qr.link",
    "buff.ly",
    "tiny.cc",
    "shor.by",
    "short.io",
    "goo.gl",
    "bl.ink",
)


def ensure_dirs() -> None:
    for d in (RESUME_DIR, CONFIG_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def assert_secrets(*names: str) -> None:
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise RuntimeError(
            f"Missing env vars: {', '.join(missing)}. Copy .env.example to .env and fill in."
        )
