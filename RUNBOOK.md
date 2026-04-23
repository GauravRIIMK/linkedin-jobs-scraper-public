# Runbook — day-to-day ops

For fresh setup, see [SETUP.md](SETUP.md). This doc covers manual ops, budget tuning, and debugging when the daily cron misbehaves.

---

## 1. Health check (one-liner)

```bash
python -m src status && python -m src diagnose && echo "ALL GREEN"
```

`status` → last few run logs + tracking counts.
`diagnose` → probes Voyage, Anthropic, Apify, Resend with tiny payloads.

If any of the three fails, the `&&` short-circuits and you get a diagnostic. This is the canary for "tomorrow's cron will work."

---

## 2. Manual runs

### Dry-run (safe, repeatable)

```bash
python -m src dry-run
```

No email, no state update. Writes preview CSV to `results/dryrun_YYYYMMDD_HHMMSS.csv` and a log to `results/run_YYYYMMDD_HHMMSS.log`. Use this to validate config changes, tune thresholds, or confirm the Apify actors are responsive before spending budget on a real run.

### Real run

```bash
python -m src run
```

Sends email, updates `config/seen_posts.json`, writes `results/matches_YYYYMMDD_HHMMSS.csv`.

### Manual apply / reject

```bash
python -m src apply "https://linkedin.com/feed/update/urn:li:activity:XYZ/"
python -m src reject "https://..."
```

Permanently hides the URL from future scans (bypasses TTL).

### Trigger a CI run

```bash
# Dry
gh workflow run daily-scan.yml -f dry_run=true

# Real
gh workflow run daily-scan.yml

# Diagnose (health probes only, no scrape)
gh workflow run daily-scan.yml -f diagnose=true

# Watch latest
gh run watch
```

---

## 3. Budget tuning

Apify free tier = **$5 compute/month**. With all defaults, a daily run costs ~$0.50–$2.00/day (mostly the paid LinkedIn posts source). Monthly worst-case: ~$60.

| Knob | Default | Cost impact | Recall impact |
| --- | --- | --- | --- |
| `MAX_POSTS_PER_QUERY` | 200 | Linear — every 100 posts ≈ $0.80/query | High |
| `QUERIES_PER_TRACK` | 10 | Linear — fewer queries = fewer posts | Medium |
| `MAX_LINKEDIN_JOBS_PER_TRACK` | 500 | Free | High |
| `MAX_NAUKRI_JOBS_PER_TRACK` | 500 | Free | High (India) |
| `ENABLED_SOURCES` | all 3 | Drop `linkedin_posts` = fully free | Loses #hiring signal |
| `ROLES_PER_JOB_BOARD_TRACK` | 4 | Linear in free-board volume | Medium |

### Stay under $5/month

```bash
gh variable set MAX_POSTS_PER_QUERY --body "50"
gh variable set QUERIES_PER_TRACK --body "5"
# Or go full free:
gh variable set ENABLED_SOURCES --body "linkedin_jobs,naukri_jobs"
```

### More matches, spend more

```bash
gh variable set MAX_POSTS_PER_QUERY --body "300"
gh variable set MAX_LINKEDIN_JOBS_PER_TRACK --body "1000"
gh variable set MAX_NAUKRI_JOBS_PER_TRACK --body "1000"
gh variable set ROLES_PER_JOB_BOARD_TRACK --body "6"
gh variable set SIMILARITY_THRESHOLD --body "0.58"
```

### Email too noisy

```bash
gh variable set LLM_SCORE_THRESHOLD --body "7.5"
gh variable set SIMILARITY_THRESHOLD --body "0.66"
```

---

## 4. Cadence tuning

Daily cron is defined in `.github/workflows/daily-scan.yml`:

```yaml
on:
  schedule:
    - cron: "30 22 * * *"    # 22:30 UTC = 04:00 IST (primary)
    - cron: "30 23 * * *"    # 23:30 UTC = 05:00 IST (backup + dedup)
```

Edit to your timezone. Notes:
- GitHub Actions cron is UTC-only and best-effort (can delay 15–60 min at peak). The backup cron exists precisely for this.
- The first step in the workflow (`Skip backup cron if primary already succeeded today`) short-circuits the backup via `gh run list` — so you never get two emails.
- If you only want one trigger, delete the `30 23 * * *` line and the `dedup` step will no-op harmlessly.

---

## 5. Priority-boost tuning

All boosts and penalties live in `src/config.py`:

```python
PRIORITY_BOOST_INDIAN_TIER1 = 0.8
PRIORITY_BOOST_USER_CITY    = 0.5
PRIORITY_BOOST_INDIA        = 0.4
PRIORITY_PENALTY_NON_INDIA  = 0.6
PRIORITY_BOOST_CONNECTION   = 0.5
PRIORITY_BOOST_RECRUITER    = 0.3
PRIORITY_BOOST_REMOTE       = 0.2
PRIORITY_BOOST_SOURCE_TIER1 = 0.2
```

Each is env-overridable (`PRIORITY_BOOST_INDIA=0.6`) so you can tune from GitHub Variables without editing code.

To expand the tier-1 Indian company list, edit `config/indian_tier1_companies.json`. To expand the Indian geo-detection, edit `_INDIA_TOKENS_SET` in `src/priority_ranker.py`.

---

## 6. Filter tuning

Two common filter-related complaints:

### "I'm still getting wrong-archetype posts"

Add a track-specific negative regex to `src/role_filter.py`:

```python
NEGATIVE_PATTERNS = {
    "ai_ops_automation": [
        r"\bsoftware engineer\b",
        r"\bSDE\b",
        # ...
        r"\byour new pattern here\b",
    ],
}
```

Add a test in `tests/test_role_filter.py` that would have caught the original miss, then run the suite.

### "I'm getting too few matches"

In order of cheap-to-expensive:

1. Raise `POSTED_LIMIT_DAYS` to 3 or 7 (wider lookback window)
2. Lower `SIMILARITY_THRESHOLD` from 0.62 to 0.58 (wider embedding funnel)
3. Lower `LLM_SCORE_THRESHOLD` from 6.5 to 6.0 (more Haiku survivors)
4. Add more `roles` aliases in `role_preferences.json` (e.g. add `Marketing Manager` alongside `Growth Marketing Manager`)
5. Add more `cities` (include every spelling variant)
6. Increase `MAX_POSTS_PER_QUERY` / `MAX_LINKEDIN_JOBS_PER_TRACK` / `MAX_NAUKRI_JOBS_PER_TRACK`

---

## 7. State management

| File | Shape | Written by |
| --- | --- | --- |
| `config/seen_posts.json` | `{urn: {url, first_seen, delivered_at}}` | Real run only |
| `config/applied_posts.json` | `{url: {marked_at}}` | `python -m src apply` |
| `config/rejected_posts.json` | `{url: {marked_at}}` | `python -m src reject` |
| `config/resume_embedding_cache.json` | `{sha256: float[1024]}` | `matcher.embed_resume` |

Dedup semantics:
- `seen_posts` entries **expire after `SEEN_TTL_DAYS` (default 3)** — a post you ignored for 3 days re-surfaces
- `applied_posts` and `rejected_posts` are **permanent** — never shown again
- `resume_embedding_cache` is keyed on content (not track name) — editing the summary invalidates the cache; editing a different track's summary doesn't

### Reset state

```bash
# Reset everything except your applied/rejected decisions
echo '{}' > config/seen_posts.json
echo '{}' > config/resume_embedding_cache.json

# Reset applied/rejected too (usually you don't want this)
echo '{}' > config/applied_posts.json
echo '{}' > config/rejected_posts.json
```

---

## 8. Common failure modes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Workflow fails on "Materialize role_preferences.json" | Secret empty or BOM-corrupted | Re-run `gh secret set ROLE_PREFERENCES_JSON < config/role_preferences.json` from bash (not PowerShell, which injects a UTF-8 BOM) |
| 0 posts from `linkedin_posts` | Apify token invalid or out of credit | `python -m src diagnose` + check Apify console |
| Voyage 429 storms | Free-tier 3 RPM / 10K TPM | Upgrade Voyage or cut `LLM_RERANK_TOP_N` |
| Email never arrives | Resend key wrong or mailbox bounced | `results/delivery_error_*.txt` + Resend dashboard |
| Cron ran but no email | Delivery failed mid-flight | Check `results/delivery_error_*.txt`, check workflow logs for `Delivery: sent=False` |
| Same posts every day | State files not persisting (public fork + no persistent state) | See [SETUP.md §6](SETUP.md#6-state-files--the-publicprivate-fork-distinction) |
| "fetch first" error on state commit | Parallel push to `main` | Workflow retries with rebase-on-remote; 3 attempts |
| Empty digest for 3+ days | Look-back too narrow OR thresholds too tight | Raise `POSTED_LIMIT_DAYS=3`, drop `LLM_SCORE_THRESHOLD=6.0` |
| Cron stopped running | GitHub disables schedules after 60 days of inactivity | Any push re-enables |

---

## 9. Running tests

```bash
python -m pytest tests/ -q
# 515 tests, ~30s on a laptop
```

Tests run on Python 3.11+ or 3.14. Voyage and Anthropic are stubbed in `tests/conftest.py` — no API calls leave the machine.

To run a specific test module:
```bash
python -m pytest tests/test_role_filter.py -v
python -m pytest tests/test_priority_ranker.py -k india -v
```

---

## 10. Log locations

On a successful run, the repo accumulates:

```
results/
├── matches_YYYY-MM-DD.csv            # Delivered matches (real runs)
├── dryrun_YYYYMMDD_HHMMSS.csv        # Dry-run previews (local only)
├── run_YYYYMMDD_HHMMSS.log           # Per-run full log
└── delivery_error_YYYYMMDD_HHMMSS.txt # Only when email delivery fails
```

The CI workflow commits `run_*.log` and `delivery_error_*.txt` back to the repo (since they're useful for post-mortems), but not `matches_*.csv` (since it contains the matched URLs and you might not want those in a public fork — adjust `.gitignore` if you're on a private fork and do want them).
