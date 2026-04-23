# Setup — end to end

This guide gets you from a fresh clone to a daily automated email digest in ~30 minutes. It assumes basic comfort with a terminal, Git, and GitHub.

If you just want to run it locally once, stop after §3.

---

## 0. Prerequisites

- **Python 3.11–3.13** (3.14 has a `pillow` dependency gap that breaks `voyageai` install; tests still pass via stub)
- **Git** and a **GitHub** account
- An **Apify** account — free tier covers the free actors; paid LinkedIn posts source needs ~$5/mo of Apify compute credit
- An **Anthropic API** key (Claude Opus for resume parsing + Claude Haiku for reranking)
- A **Voyage AI** API key (embeddings)
- A **Resend** account OR Gmail with 2FA enabled (for email delivery)

All four API services have free tiers. Expected monthly cost with defaults: ~$15–$30 if `linkedin_posts` source is enabled, ~$0.20 if only the free sources are used.

---

## 1. Clone + local install

```bash
git clone https://github.com/<you>/linkedin-jobs-scraper
cd linkedin-jobs-scraper
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python -m pytest tests/ -q       # sanity check — 515 tests should pass
```

---

## 2. Configure your preferences

### 2a. Copy the example config

```bash
cp .env.example .env
cp config/role_preferences.example.json config/role_preferences.json
```

### 2b. Fill in `.env`

Open `.env`, fill in your API keys. The file already documents each key's purpose and where to get it.

Minimum required for a local run:
```
APIFY_TOKEN=apify_api_...
ANTHROPIC_API_KEY=sk-ant-api...
VOYAGE_API_KEY=pa-...
EMAIL_TO=your-email@gmail.com
DELIVERY_METHOD=resend
RESEND_API_KEY=re_...
```

### 2c. Edit `config/role_preferences.json`

This is the single most important configuration. Each "track" is a parallel resume-lens:

```json
{
  "tracks": [
    {
      "track": "growth_marketing",
      "active": true,
      "roles": ["Growth Marketing Manager", "Head of Growth", "Growth Manager"],
      "cities": ["Bangalore", "Bengaluru"],
      "include_remote": true,
      "resume_summary": "Growth marketer with 6+ years across SaaS / fintech. Skills: lifecycle, CRM, cohort analytics. Past roles at [company A], [company B]. Looking for senior manager or director roles at Series B+ startups. Not looking for: performance marketing / paid-media / SEO-SEM roles."
    }
  ]
}
```

Fields per track:

- **`track`** — slug, no spaces. Used as filename matcher for resume PDFs (e.g. `growth_marketing.pdf` → `growth_marketing` track).
- **`active`** — set `false` to skip this track this cycle.
- **`roles`** — 3–5 exact job titles recruiters use. Drive LinkedIn search queries; be specific (`Growth Marketing Lead` matches, `growth-minded marketer` doesn't).
- **`cities`** — include both local spellings (`Bangalore` AND `Bengaluru`). Each is queried separately.
- **`include_remote`** — adds `"<role>" hiring remote` queries.
- **`resume_summary`** — ~200 words of your actual background. **This is the text the pipeline semantically compares to every job post.** Be honest and dense. Skills, years, industries, notable wins. Don't keyword-stuff; the reranker is an LLM. Include an explicit **"Not looking for:"** line listing the sub-specializations you want excluded — the LLM reranker weighs this heavily.

Save. Re-run tests to confirm the JSON is valid:
```bash
python -m pytest tests/test_config.py -q
```

### 2d. (Optional) Drop PDF resumes for auto-parse

If you'd rather not hand-write `resume_summary`, put one PDF per track in `./resume/` (filename = `<track_slug>.pdf`) and run:

```bash
python -m src setup
```

This uses Claude Opus to parse each PDF into structured data, then generates a starting `resume_summary`. You'll still want to hand-edit the summary afterwards to refine it.

---

## 3. Local dry-run → real run

```bash
# Dry run: scrapes, filters, ranks, but NO email, NO state update
python -m src dry-run

# Check the preview CSV in results/
# If counts look reasonable, go live:
python -m src run
```

A successful run ends with `Delivery: sent=True`. The email lands in your inbox within ~60 seconds. If it lands in Spam, mark as Not Spam to train Gmail.

**CLI reference:**
```
python -m src setup                 Parse resumes + confirm roles (interactive)
python -m src run                   Full scan + send email + mark seen
python -m src dry-run               Same as run but no email, no state update
python -m src status                Recent logs + tracking stats
python -m src apply <url>           Mark post as applied (hides it permanently)
python -m src reject <url>          Mark post as rejected (hides it permanently)
python -m src tracks                Print current track config
python -m src tracks --toggle <t>   Activate/deactivate a track
python -m src diagnose              API health probes for all 4 services
```

---

## 4. GitHub Actions automation

Two daily crons with a dedup guard, pre-wired at `.github/workflows/daily-scan.yml`. **Free for public repos**, so hosting this here costs $0 in GitHub minutes.

### 4a. Push to GitHub

```bash
git remote add origin git@github.com:<you>/<repo>.git
git push -u origin main
```

### 4b. Configure repo Secrets

Use the `gh` CLI (or go to Settings → Secrets and variables → Actions → Secrets):

```bash
gh secret set APIFY_TOKEN --body "apify_api_..."
gh secret set ANTHROPIC_API_KEY --body "sk-ant-api..."
gh secret set VOYAGE_API_KEY --body "pa-..."
gh secret set RESEND_API_KEY --body "re_..."
gh secret set EMAIL_TO --body "your-email@gmail.com"

# Role preferences must come from your local file:
gh secret set ROLE_PREFERENCES_JSON < config/role_preferences.json
```

(If you use Gmail SMTP instead of Resend, also set `GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD`, and set the `DELIVERY_METHOD` variable to `"gmail"`.)

### 4c. (Optional) Configure tunable Variables

Variables are non-secret knobs. Defaults in `.env.example` are safe — skip this section unless you want to tune.

```bash
gh variable set POSTED_LIMIT_DAYS --body "2"           # lookback window
gh variable set MAX_POSTS_PER_QUERY --body "200"       # paid LinkedIn cap
gh variable set QUERIES_PER_TRACK --body "10"
gh variable set ENABLED_SOURCES --body "linkedin_posts,linkedin_jobs,naukri_jobs"
gh variable set SIMILARITY_THRESHOLD --body "0.62"
gh variable set LLM_SCORE_THRESHOLD --body "6.5"
```

### 4d. Trigger a manual run to verify

```bash
# Dry run via UI: Actions → Daily LinkedIn Scan → Run workflow → dry_run=true
# Or via CLI:
gh workflow run daily-scan.yml -f dry_run=true
gh run watch                                        # follow the log
```

Look for these log lines:
- `Pipeline start · dry_run=True`
- `<track>: N raw posts` across all tracks (zero across the board = your roles/cities are too narrow)
- `Match summary` table at the end

If the dry run looks good, fire a real one:
```bash
gh workflow run daily-scan.yml        # defaults to real run
```

Wait for the email. Check `results/run_*.log` in the repo after the run — it's committed back by the workflow.

### 4e. The two crons

```yaml
- cron: "30 22 * * *"   # 04:00 IST (primary)
- cron: "30 23 * * *"   # 05:00 IST (backup — dedups if primary succeeded)
```

Edit to your own timezone. GitHub Actions runs in UTC. The first step in `daily-scan.yml` short-circuits the backup if the primary already succeeded in the last 3 hours, so you never get two emails on the same day.

---

## 5. Updating your preferences later

Edit `config/role_preferences.json` locally, then:

```bash
gh secret set ROLE_PREFERENCES_JSON < config/role_preferences.json
# next daily run picks up the change automatically
```

No re-deploy, no workflow edit.

---

## 6. State files & the public/private fork distinction

State files live at:
- `config/seen_posts.json` — URLs already emailed (TTL-aware, 3-day default)
- `config/applied_posts.json` — URLs you applied to
- `config/rejected_posts.json` — URLs you rejected
- `config/resume_embedding_cache.json` — cached resume embeddings

**In this public repo**, they're gitignored. Each run starts fresh from an empty cache on the GitHub runner.

**If you want state to persist across CI runs** — the workflow has a commit-back step that pushes them to the repo. For that to work, you need to:

1. Fork into a **private** repo (don't expose your scrape history publicly)
2. Delete the four lines under `# Runtime state (private per user)` in `.gitignore`
3. Push — from then on the workflow commits `seen_posts.json` etc. back after each run

Without this, every run re-scrapes from scratch and you'll see duplicates of posts from previous days that still match.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Workflow fails on "Materialize role_preferences.json" | `ROLE_PREFERENCES_JSON` secret empty or malformed | `gh secret set ROLE_PREFERENCES_JSON < config/role_preferences.json` |
| 0 posts from `linkedin_posts` source | Apify token invalid or out of credit | `python -m src diagnose` |
| Voyage rate-limit storms | Free-tier 3 RPM / 10K TPM ceiling | Upgrade Voyage or lower `LLM_RERANK_TOP_N` |
| Email never arrives | Resend key wrong or `EMAIL_TO` bounced | Check `results/delivery_error_*.txt`; Resend dashboard shows bounces |
| Same posts every day | State files not persisting (public fork) | See §6 — private fork + remove .gitignore lines |
| Empty digest several days running | Lookback too narrow OR Haiku threshold too tight | Raise `POSTED_LIMIT_DAYS` to 3, drop `LLM_SCORE_THRESHOLD` to 6.0 |
| Cron stopped running | GitHub disables scheduled workflows after 60 days of repo inactivity | Any push re-enables |

More debugging + ops tips: [RUNBOOK.md](RUNBOOK.md).

---

## 8. Verified end-to-end

On a clean clone with real API keys, a real run completes in ~40 minutes:
- ~8,000 raw posts scraped across 4 tracks
- ~5,500 after apply-method filter
- ~4,000 after YoE filter
- ~3,200 after role-archetype filter
- 120 top-K by cosine (30/track)
- 10–15 survive Haiku ≥ 6.5 threshold
- Email delivered with HTML + CSV

You're good to go.
