# LinkedIn Jobs Scraper

> A personal-scale daily job-discovery pipeline for **non-SWE job seekers**. Scrapes LinkedIn #hiring posts, LinkedIn Jobs, and Naukri across multiple resume tracks; filters by apply-method, experience band, and role archetype; ranks semantically against your resume; emails you the top handful every morning.

Built for one user (the author), released publicly under MIT. Fork it, point it at your own resume, and it works for you too.

---

## Why

Most job-search tooling assumes you're a software engineer. If you're a growth marketer, product manager, strategy consultant, or AI/ops generalist, the signal-to-noise on LinkedIn is brutal: ~3,000 relevant-looking postings/day for any given track, most of which are wrong-level, wrong-geo, or actually for coders with "AI" in the title.

This pipeline turns that 3,000 into **5–15 high-fit daily matches** in an email, at roughly **~$4/day** of API cost (or ~$0/day if you skip the paid LinkedIn posts source).

## What it does

Every morning at your chosen hour:

1. **Fans out to 3 Apify scrapers** in parallel (LinkedIn #hiring posts — paid signal; LinkedIn Jobs board — free; Naukri.com — free, India-heavy)
2. **Filters** by apply method (Google Forms, 19 ATS platforms, LinkedIn Easy Apply, mail-to-apply), years-of-experience cap, and a role-archetype gate (blocks SWE titles from a PM track, blocks paid-media titles from a brand track, etc.)
3. **Embeds** each post against your resume summary with Voyage-3 and keeps the top-K by cosine similarity
4. **Reranks** the shortlist with Claude Haiku (0–10 fit score + one-line reason)
5. **Adds priority boosts** — tier-1 Indian companies, user's home city, India-located roles, tier-1 source, verified recruiter, connection-at-company
6. **Deduplicates** across all tracks so no post appears twice, assigns each to its best-fit track
7. **Emails** an HTML digest with per-track sections + a CSV attachment, via Resend or Gmail SMTP

State (seen URLs, applied/rejected lists, embedding cache) lives in JSON files under `config/`. Dedup is TTL-aware — a post re-appears 3 days after it's first seen if you didn't act on it.

## Pipeline at a glance

```
          ┌───────────────────────────────────────────────────────────────┐
          │ role_preferences.json  (tracks × roles × cities × resume_summary)
          └─────────────┬─────────────────────────────────────────────────┘
                        │
                        ▼
         ┌────────────────────────────────┐
         │ 1. Scrape  (Apify fan-out)     │    3 sources × ~4 tracks
         │   • harvestapi LinkedIn posts  │    = 8,000–15,000 raw posts/day
         │   • LinkedIn Jobs (free)       │
         │   • Naukri Jobs (free, IN)     │
         └────────────────┬───────────────┘
                          ▼
         ┌────────────────────────────────┐    drops ~70% — keeps posts with
         │ 2. Application filter          │    a gform / ATS / Easy Apply /
         │    + TTL dedup vs seen_posts   │    mail / structured apply URL
         └────────────────┬───────────────┘
                          ▼
         ┌────────────────────────────────┐    keeps only roles ≤ your YoE
         │ 3. Experience filter (YoE)     │    + a buffer (default ≤ 7 yrs)
         └────────────────┬───────────────┘
                          ▼
         ┌────────────────────────────────┐    title-window regex + body-
         │ 4. Role-archetype gate         │    declarative scan blocks
         │    (blocks wrong-archetype)    │    coder titles from non-coder
         └────────────────┬───────────────┘    tracks etc.
                          ▼
         ┌────────────────────────────────┐    cosine ≥ 0.62 (default)
         │ 5. Voyage-3 embedding match    │    keeps top-30 per track
         └────────────────┬───────────────┘
                          ▼
         ┌────────────────────────────────┐    score ≥ 6.5 (default)
         │ 6. Claude Haiku rerank         │    fit reason: "Strong match —
         │    (0-10 + one-line reason)    │    growth marketing at fintech"
         └────────────────┬───────────────┘
                          ▼
         ┌────────────────────────────────┐    Indian tier-1, home city,
         │ 7. Priority boost + dedup      │    India location, recruiter,
         │    (best-fit track assignment) │    connection, remote, source
         └────────────────┬───────────────┘
                          ▼
         ┌────────────────────────────────┐    HTML + CSV attachment,
         │ 8. Email delivery              │    via Resend or Gmail SMTP
         └────────────────────────────────┘
```

**Detailed architecture:** [ARCHITECTURE.md](ARCHITECTURE.md) — module-by-module walkthrough, filter funnel math, dedup semantics, and design tradeoffs.

## Quick start (local)

Requires Python 3.11+ (tested on 3.11–3.13; 3.14 has a `pillow` wheel gap that breaks `voyageai` install — tests still pass via stub).

```bash
git clone https://github.com/<you>/linkedin-jobs-scraper
cd linkedin-jobs-scraper
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

cp .env.example .env            # then edit .env with your API keys
cp config/role_preferences.example.json config/role_preferences.json
# edit config/role_preferences.json — it's your resume summary + target roles

# Drop PDF resumes in ./resume/ (one per track), or skip and hand-write
# the resume_summary field in role_preferences.json

python -m src setup             # interactive: parses resumes + confirms roles
python -m src dry-run           # runs the pipeline, no email, no state update
python -m src run               # real run — sends email + updates seen state
```

**Full step-by-step setup:** [SETUP.md](SETUP.md).
**Daily ops, tuning, and troubleshooting:** [RUNBOOK.md](RUNBOOK.md).

## Weekly/daily automation

GitHub Actions workflow is pre-wired at `.github/workflows/daily-scan.yml`:

- Two crons (primary + backup) with a dedup step so the backup no-ops if the primary already succeeded
- All config flows in via repo Secrets + Variables — zero code changes to tune
- State files (`seen_posts.json` etc.) are *gitignored* in this public repo — if you fork into a **private** repo and want state to persist across CI runs, delete those lines from `.gitignore` so the workflow can commit them back

See [SETUP.md §4](SETUP.md#4-github-actions-automation) for the full CI setup.

## Cost envelope

Per day, all 3 sources active, 4 tracks, default caps:

| Service                     | Unit                  | Daily usage        | Cost    |
| --------------------------- | --------------------- | ------------------ | ------- |
| Apify HarvestAPI (LinkedIn) | $0.002 / post         | ~2,000 posts       | ~$4.00  |
| Apify LinkedIn Jobs         | Free actor            | ~2,000 listings    | $0      |
| Apify Naukri                | Free actor            | ~1,500 listings    | $0      |
| Voyage-3 embeddings         | $0.06 / M tokens      | ~50k tokens        | ~$0.003 |
| Claude Haiku rerank         | $0.25 / M input       | ~30k tokens        | ~$0.008 |
| Claude Opus (resume parse)  | $15 / M input         | once per resume ed | ~$0.20  |
| Resend email                | Free tier (100/day)   | 1 email            | $0      |
| GitHub Actions              | Free for public repos | ~40 min/day        | $0      |
| **Total**                   |                       |                    | **~$4/day** |

To cut cost to ~$0/day, drop `linkedin_posts` from `ENABLED_SOURCES` — you lose the paid #hiring-post signal but keep both free boards.

## Security & privacy

- **No API keys in the repo.** `.env` is gitignored. Example file uses placeholder strings.
- **No personal data in the repo.** `config/role_preferences.example.json` uses placeholder text. Real preferences (`config/role_preferences.json`) are gitignored; in CI they flow through the `ROLE_PREFERENCES_JSON` secret.
- **No scraped URLs in the repo.** State files (`seen_posts.json`, `applied_posts.json`, `rejected_posts.json`, `resume_embedding_cache.json`) are gitignored for the public fork. Private-fork users who want state persistence across CI runs should remove those lines from `.gitignore`.
- **Resume PDFs never leave your machine.** They're parsed locally (or in your own CI job). `resume/*.pdf` is gitignored.
- **Resend API keys can only send email** — a leaked Resend key cannot read your inbox. Gmail app passwords are more sensitive; prefer Resend if you're exposing the repo.
- **GitHub Actions secrets are encrypted at rest** and redacted from logs by the runner.

## Tests

```bash
python -m pytest tests/ -q
# 515 tests pass on a clean checkout
```

Voyage and Anthropic are stubbed in tests — no network calls leave the machine.

## Contributing / forking

This is a personal project the author runs daily. PRs for generic improvements (new source adapters, additional ATS detectors, better filter heuristics) are welcome. For track-specific or geography-specific tweaks, fork and tune locally.

## License

MIT — see [LICENSE](LICENSE).
