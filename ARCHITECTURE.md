# Architecture

This document walks through the pipeline stage-by-stage. Read `README.md` first for the high-level picture; this file covers module boundaries, data shapes, and why each filter exists.

---

## The mental model

The pipeline is a **funnel over LinkedIn-quality noise**. Each stage drops posts; each stage is independently tunable; each stage fails soft (one track or one source blowing up doesn't block the others).

Rough per-day volumes observed in production (4 active tracks, all 3 sources):

| Stage                    | Posts remaining | Drop mechanism               |
| ------------------------ | --------------- | ---------------------------- |
| Raw scrape               | ~8,000          | Apify actors                 |
| After URL dedup          | ~7,000          | Cross-source URL normalize   |
| After apply-method       | ~4,500          | `application_filter`         |
| After TTL-aware dedup    | ~4,000          | `tracking.drop_seen`         |
| After YoE cap            | ~3,200          | `experience_filter`          |
| After role archetype     | ~2,900          | `role_filter`                |
| Top-K by cosine (per tr) | 30/track = 120  | `matcher` + Voyage embeds    |
| After Haiku threshold    | 10–20           | `reranker`                   |
| After cross-track dedup  | 5–15            | `reranker.assign_best_track` |

Order matters: cheap/deterministic filters run before expensive/LLM stages.

---

## Stage 1 · Scrape (`src/sources/`)

**Input:** `role_preferences.json` + a `queries_by_track` map produced by `query_builder`.

**Output:** `posts_by_track: dict[track_slug, list[post_dict]]`

Three adapters, all behind a uniform contract (`scrape(prefs, queries, track) → list[post]`). The orchestrator (`src/sources/__init__.py::scrape_all`) fans them out per track and URL-dedupes the combined stream.

- **`linkedin_posts.py`** — wraps Apify's `harvestapi/linkedin-post-search` actor. Best-signal source because it sees the full #hiring feed with author/company metadata. Paid (~$0.002/post). Fed the full user resume summary as `postedLimitBucket` biasing.
- **`linkedin_jobs.py`** — wraps `worldunboxer/rapid-linkedin-scraper`. Free. Reads the structured LinkedIn Jobs board. Less noisy than posts but less coverage of startup/recruiter-posted openings.
- **`naukri_jobs.py`** — wraps `shahidirfan/Naukri-Jobs-Scraper`. Free. India-heavy, complements LinkedIn well for Indian companies not active on LinkedIn.

**Scrape soft deadline** — configurable via `SCRAPE_SOFT_DEADLINE_SECONDS` (default 2400). When the clock passes this, the orchestrator stops starting *new* tracks and hands whatever it has to the downstream stages. This matters because GitHub Actions has a hard `timeout-minutes` ceiling; without the soft deadline, a slow Apify day would hit the hard limit and throw everything away mid-pipeline.

**Fail-soft semantics** — each source wraps its actor call in try/except; a failing actor logs a warning and returns `[]` rather than propagating the exception. The orchestrator's per-track summary (`linkedin_post=1124, linkedin_job=429, naukri=493`) makes debugging trivial.

---

## Stage 2 · Application filter (`src/application_filter.py`)

Purpose: drop anything you can't actually apply to without clicking through a maze.

Detects apply methods in this priority order:

1. **`gform`** — Google Forms. Scans post body + all comments + nested replies, follows up to 5 redirect hops through 20+ shortener hosts (`lnkd.in`, `bit.ly`, custom domains), parses landing-page HTML for hidden `forms.gle` / `docs.google.com/forms` links, and handles `<meta http-equiv="refresh">` redirects.
2. **`ats`** — 19 ATS platforms: Lever, Greenhouse, Ashby, Workable, Teamtailor, Recruitee, Breezy, Dover, Wellfound, JobVite, SmartRecruiters, Polymer, Rippling, Gusto, BambooHR, Freshteam, Keka, Zoho Recruit, Darwinbox.
3. **`linkedin_easy_apply`** — flag on the post tells us there's a LinkedIn Easy Apply form available.
4. **`linkedin_job`** — a standard LinkedIn Jobs board listing (from the `linkedin_jobs` source).
5. **`naukri`** — Naukri.com listings always have an apply button (from the `naukri_jobs` source).
6. **`mail`** — `apply to foo@bar.com` or `dm me at foo@bar.com` detected. Mailtoes work too.
7. **`soft_dm`** — fallback: "DM me to apply" / "comment interested" phrasing. Lowest priority; can be toggled off in `config.ENABLED_APPLY_METHODS`.

Posts without any detected apply method are dropped. This is the biggest single funnel — typically ~40–60% of raw posts are general-update-not-really-hiring noise.

**TTL-aware dedup** — right after apply-method, the pipeline drops posts in `seen_posts.json` whose `delivered_at` is less than `SEEN_TTL_DAYS` (default 3) old. A post delivered 4 days ago and not-acted-on re-surfaces. Posts in `applied_posts.json` or `rejected_posts.json` are dropped permanently regardless of TTL.

---

## Stage 3 · Experience filter (`src/experience_filter.py`)

Parses the YoE requirement out of the JD body using regex patterns for the common shapes:

- `"5+ years"`, `"5-8 years"`, `"5 to 8 years"`
- `"minimum 5 years"`, `"at least 5 years"`
- `"5yrs experience"`, `"5 yr exp"`

Drops posts where the **lower bound** exceeds `USER_YEARS_EXPERIENCE + EXPERIENCE_BUFFER_YEARS` (defaults: 4.5 + 2.5 = 7 years). The buffer is deliberately generous — recruiters overstate, the LLM reranker is a better judge of actual fit.

Posts with no parseable YoE requirement pass through untouched.

---

## Stage 4 · Role archetype gate (`src/role_filter.py`)

The hard one. LinkedIn semantically overlaps "AI Product Manager" and "ML Engineer" despite those being completely different archetypes. This filter blocks the specific wrong-archetype leakage per track.

**Two detection modes:**

1. **Title-window scan** — grabs the first 480 chars of `author_title` / `job_title` / `role` / `headline`, strips common framing prefixes (`[post body]`, `[from comments]`), and runs a list of track-specific negative regexes. Example negatives for `ai_ops_automation`: `software engineer`, `SDE`, `full[\s-]?stack developer`, `ML engineer`, `data scientist`, `devops`, `SRE`.
2. **Body-declarative scan** — scans the first 2,000 chars of JD body for recruiter-preamble-then-role-declaration patterns: `hiring a Digital Marketing Manager`, `Role: SEO Specialist`, `Position: Software Engineer`. Catches the common pattern where the author_title is empty (it's a recruiter's general network post) and the real role is buried under boilerplate.

**Positive rescue** — if the title hits a negative but also matches a track-specific positive (e.g. `Automation Specialist - Python Developer scripting needed` hits `python developer` in negatives but is rescued by `automation specialist` in positives), it passes. Positives have higher precedence than negatives.

**Tracks without negatives** (e.g. `strategy_consulting`, `product_brand_category`) let everything through at this stage and lean on the embedding + LLM reranker downstream.

---

## Stage 5 · Voyage-3 embedding match (`src/matcher.py`)

For each surviving post:

1. Concatenate `extract_jd(post)` → 400–2000 chars covering body + comment replies + live-fetched ATS page content
2. Embed with Voyage-3 (1024-dim)
3. Cosine-compare against the track's `resume_summary` embedding
4. Keep top-N above `SIMILARITY_THRESHOLD` (default 0.62, top-30 by default)

**Resume embedding cache** — `config/resume_embedding_cache.json` stores `sha256(resume_summary) → vector` so editing preferences doesn't re-embed the resume on every run. Cache entry is keyed on the summary content, not the track name, so two tracks with the same summary share one embedding.

**JD extractor** — `src/jd_extractor.py` normalizes the JD text. Merges post body + top-level comments + nested replies into one embedding-friendly string. Live-fetches JD pages for ATS/Notion/company-career URLs using BeautifulSoup with readability-style extraction (prefers `<main>` / `<article>`). Flags private Google Docs / Drive links as click-through (no auth = no fetch). Rejects LinkedIn-internal and social-media URLs as noise.

---

## Stage 6 · Claude Haiku rerank (`src/reranker.py`)

Top-30 posts per track go to Claude Haiku (currently `claude-haiku-4-5`) with a prompt containing:

- Full `resume_summary` for the track
- JD text (truncated to ~2,500 chars)
- Request for a 0–10 fit score + one-line reason

Posts below `LLM_SCORE_THRESHOLD` (default 6.5) are dropped. Surviving posts keep the Haiku score and reason — the score drives ordering in the final email, the reason is rendered as a subtitle next to the role.

The reranker is the most expensive stage per-post but it only sees the top-30 that survived the cheaper filters. Keeps the bill under $0.01/day.

---

## Stage 7 · Priority boost + cross-track dedup (`src/priority_ranker.py` + `src/reranker.py::assign_best_track`)

**Priority boost** — after Haiku scoring, `compute_priority_boost` adjusts the score in-place with additive deltas. Each boost is gated so they don't double-count (e.g. `tier1 ⊃ user_city ⊃ india`). Current boosts:

| Signal                              | Magnitude | Config knob                        |
| ----------------------------------- | --------- | ---------------------------------- |
| Tier-1 Indian company               | +0.8      | `PRIORITY_BOOST_INDIAN_TIER1`      |
| User's home city (e.g. Gurgaon)     | +0.5      | `PRIORITY_BOOST_USER_CITY`         |
| India-located (structured / body)   | +0.4      | `PRIORITY_BOOST_INDIA`             |
| Non-India location penalty          | −0.6      | `PRIORITY_PENALTY_NON_INDIA`       |
| Connection at company (LinkedIn CSV) | +0.5      | `PRIORITY_BOOST_CONNECTION`        |
| Recruiter/verified badge            | +0.3      | `PRIORITY_BOOST_RECRUITER`         |
| Remote-friendly                     | +0.2      | `PRIORITY_BOOST_REMOTE`            |
| Tier-1 source (linkedin_posts)      | +0.2      | `PRIORITY_BOOST_SOURCE_TIER1`      |

Each applied boost is appended to a `boost_reasons` list and rendered as a chip badge in the final email (e.g. `[Indian Tier-1] [Gurgaon] [Recruiter]`).

**India detection** — two-pass:

1. Structured location field match against `_INDIA_TOKENS_SET` (~25 city/state tokens — Mumbai, Bangalore, Delhi, Pune, Hyderabad, Chennai, Gurgaon, Gurugram, Noida, Kolkata, Ahmedabad, Jaipur, Indore, Kerala, Karnataka, Maharashtra, etc.)
2. Body preamble scan — `"based in"`, `"location:"`, `"office in"`, `"headquartered in"`, `"hq in"`, `"posted in"` + the next ~80 chars, plus a standalone `pan-india` check

Casual body mentions like `"customers in India"` do NOT fire the boost — that was a known false-positive and the scan is deliberately narrow.

**Cross-track dedup** — if a post matches multiple tracks (e.g. "AI Product Manager" matches `ai_ops_automation` AND `product_brand_category`), `assign_best_track` picks the track with the highest post-boost score and drops the duplicates. No post appears in two email sections.

---

## Stage 8 · Delivery (`src/delivery.py`)

- **HTML email** — per-track sections, each post rendered as a card with title, company, city, chip badges, fit reason, apply button, and a "why this matched" expandable.
- **CSV attachment** — flat table with all fields for follow-up in a spreadsheet.
- **Two backends** — Resend (default, just an API key, ~3,000 free emails/mo) and Gmail SMTP (needs 2FA + app password).

Failure modes are caught and written to `results/delivery_error_YYYYMMDD_HHMMSS.txt` which the CI workflow commits back — so the repo has a history of delivery health.

---

## State & dedup

Four files under `config/`:

- **`seen_posts.json`** — `{urn: {url, first_seen, delivered_at}}`. Written after every real run; the TTL-aware dedup at stage 2 reads it.
- **`applied_posts.json`** — permanently-hidden URLs. Populated manually via `python -m src apply <url>`.
- **`rejected_posts.json`** — permanently-hidden URLs. Populated manually via `python -m src reject <url>`.
- **`resume_embedding_cache.json`** — `sha256(resume_summary) → float[1024]`. Keyed on content not track name.

**These are gitignored** in this public repo. If you fork into a private repo and want state to persist across GitHub Actions runs, remove the four lines under `# Runtime state` from `.gitignore` — the included workflow has a commit step that pushes them back.

---

## Configuration surface

All tunable knobs live in `src/config.py` and are env-overridable. The workflow file exposes the important ones through `${{ vars.FOO || 'default' }}` so you can tune from the GitHub UI without touching code. See `.env.example` for the full set.

Key knobs by frequency of tweaking:

- `SIMILARITY_THRESHOLD` (0.62) — wider funnel vs tighter; drop to 0.58 for more matches, raise to 0.66 for less noise
- `LLM_SCORE_THRESHOLD` (6.5) — minimum Haiku score to survive into email; raise to 7.5 to cut noise
- `POSTED_LIMIT_DAYS` (2) — look-back window for Apify; 1 = only today, 7 = past week
- `MAX_POSTS_PER_QUERY` (200) — linear cost lever on the paid LinkedIn posts source
- `ENABLED_SOURCES` — drop `linkedin_posts` to go free-tier

---

## Why this shape

A few non-obvious choices:

**Multi-track over single-track.** Non-SWE job seekers have overlapping candidate archetypes (growth marketer ≈ brand manager ≈ product marketer). A single combined resume summary either over-matches (too wide) or under-matches (too narrow). Three separate summaries with per-track roles lets the filter + ranker optimize for each archetype independently.

**TTL dedup over permanent dedup.** A recruiter posts, you miss it, you don't apply, three days later the post is still open and you should see it again. Permanent dedup buries it; TTL respects human decision latency.

**Cheap filters before LLM.** Apply-method + YoE + role-archetype filters are regex/JSON lookups. They're effectively free and drop 75% of volume. The LLM reranker only sees top-30 per track — ~$0.01/day total.

**Fail-soft per track and per source.** One source's outage or one track's regex bug can't break the daily email. Partial data > no data.

**GitHub Actions cron with dedup guard.** A single cron occasionally gets delayed by GitHub's scheduler (15–60 min at peak). Two crons (primary + 1hr backup) plus a `gh run list` dedup check means the email arrives ~99% of days without ever sending twice.

---

## File tree

```
src/
├── __init__.py / __main__.py / cli.py     CLI entrypoints and subcommand dispatch
├── config.py                              Env-backed knobs (~50 tunable constants)
├── resume_analyzer.py                     Claude Opus → structured profile from PDF
├── role_selector.py                       questionary + rich HITL for role confirmation
├── query_builder.py                       Role × city × intent → LinkedIn search queries
├── apify_runner.py                        Low-level Apify actor caller
├── sources/
│   ├── __init__.py                        Multi-source fan-out + URL dedup + stats
│   ├── linkedin_posts.py                  harvestapi/linkedin-post-search (paid)
│   ├── linkedin_jobs.py                   worldunboxer/rapid-linkedin-scraper (free)
│   └── naukri_jobs.py                     shahidirfan/Naukri-Jobs-Scraper (free)
├── application_filter.py                  gform / ATS / Easy Apply / mail / soft_dm detection
├── experience_filter.py                   YoE requirement parser + cap enforcement
├── role_filter.py                         Title-window + body-declarative archetype gate
├── gform_filter.py                        Google Form URL detection + shortener hopping
├── jd_extractor.py                        JD text normalization + live ATS fetch
├── matcher.py                             Voyage-3 embed + cosine + resume cache
├── reranker.py                            Claude Haiku scoring + best-track assignment
├── priority_ranker.py                     Post-Haiku additive boosts (tier-1, city, India, etc.)
├── tracking.py                            seen / applied / rejected state (TTL-aware)
├── delivery.py                            Resend / Gmail SMTP + HTML/CSV rendering
├── diagnose.py                            API health probes (Voyage / Anthropic / Apify / Resend)
└── pipeline.py                            End-to-end orchestrator (fail-soft per track)

config/
├── role_preferences.example.json          Template (gitignored version is your real one)
├── linkedin_connections.example.csv       LinkedIn-exported "Connections" CSV (anonymized)
└── indian_tier1_companies.json            Curated Indian unicorns/soonicorns list

.github/workflows/
└── daily-scan.yml                         Primary + backup cron with dedup guard

tests/
└── 515 passing unit + integration tests   Network calls stubbed via tests/conftest.py
```
