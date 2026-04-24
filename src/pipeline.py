"""End-to-end orchestrator: scrape → filter → match → rerank → dedup → deliver."""
from __future__ import annotations

import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from rich.console import Console
from rich.table import Table

from . import (
    application_filter,
    config,
    delivery,
    experience_filter,
    jd_extractor,
    matcher,
    priority_ranker,
    query_builder,
    reranker,
    role_filter,
    role_selector,
    sources,
    tracking,
)

console = Console()


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _log(log: list[str], msg: str) -> None:
    line = f"[{_stamp()}] {msg}"
    log.append(line)
    console.print(line)


def run_pipeline(dry_run: bool = False) -> dict[str, Any]:
    """Execute one full daily scan. Fail-soft: one track's failure doesn't block others.

    Stage order (each stage fails soft per-track):
      1. Scrape multi-source → raw posts
      2. Filter: (a) apply-method present, (b) extract JD text, (c) drop seen
                 (TTL-aware), (d) YoE ≤ user+buffer, (e) role-archetype fit
      3. Match: Voyage embedding similarity, keep top-N
      4. Rerank: Claude Haiku score vs resume_summary
      5. Priority boost: source / company / remote / city adjustments
      6. Cross-track dedup: assign each post to its best-fit track
      7. Record seen (stamped with delivered_at for 3-day TTL) & deliver
    """
    config.ensure_dirs()
    log: list[str] = []
    _log(log, f"Pipeline start · dry_run={dry_run}")

    prefs = role_selector.load_preferences()
    queries_by_track = query_builder.build_all_queries(prefs)
    if not queries_by_track:
        _log(log, "No active tracks — nothing to do. Re-run `setup` to activate.")
        return {"matches": {}, "log": log}

    _log(log, f"Active tracks: {list(queries_by_track.keys())}")
    for track, queries in queries_by_track.items():
        _log(log, f"  {track}: {len(queries)} queries")

    # 1. Scrape — fan out to every enabled source (LinkedIn posts + LinkedIn
    #    jobs + Naukri jobs by default). The orchestrator dedupes by URL
    #    inside each track and fails soft: one source's outage never blocks
    #    the others. See src/sources/__init__.py for the source contract.
    _log(
        log,
        f"Scraping sources={config.ENABLED_SOURCES} via Apify …",
    )
    try:
        posts_by_track = sources.scrape_all(prefs, queries_by_track)
    except Exception as exc:
        _log(log, f"FATAL scrape error: {exc}")
        _save_log(log)
        return {"matches": {}, "log": log, "error": str(exc)}

    source_dist = sources.source_distribution(posts_by_track)
    for t, ps in posts_by_track.items():
        dist = source_dist.get(t, {})
        dist_str = ", ".join(f"{k}={v}" for k, v in sorted(dist.items())) or "none"
        _log(log, f"  {t}: {len(ps)} raw posts ({dist_str})")

    # 2. Filter pipeline (cheap → expensive ordering):
    #    (a) apply-method present (gform, 19 ATS platforms, LinkedIn Easy
    #        Apply, or mail-to-apply)
    #    (b) extract merged JD (body + comments + fetched doc text)
    #    (c) drop seen (TTL window = config.DEDUP_DAYS, so 3-day-old
    #        deliveries come back into rotation)
    #    (d) YoE filter: reject JDs asking for significantly more YoE than
    #        the user has. Prevents "10+ years required" roles from wasting
    #        Voyage/Haiku API spend downstream.
    #    (e) Role-archetype TAG (soft mode, not drop): each post gets a
    #        `role_filter_verdict` of "pass" or "off_archetype". Off-archetype
    #        posts still flow through embedding + Haiku + boosts — they just
    #        carry a configurable penalty (config.PRIORITY_PENALTY_OFF_ARCHETYPE)
    #        at the priority-ranker stage. This honors the user's directive
    #        that all leads get analyzed; only posts with weak fit AND weak
    #        boosts end up filtered by the final threshold.
    #        Extend role_filter.TRACK_NEGATIVES as new archetype drift shows up.
    filtered_by_track: dict[str, list[dict[str, Any]]] = {}
    for track, posts in posts_by_track.items():
        with_apply = application_filter.filter_posts_with_apply_method(posts)
        dist = application_filter.method_distribution(with_apply)
        dist_str = ", ".join(f"{k}={v}" for k, v in sorted(dist.items())) or "none"
        enriched = [{**p, **jd_extractor.extract_jd(p)} for p in with_apply]
        fresh = tracking.drop_known(enriched)
        yoe_kept, yoe_dropped = experience_filter.filter_by_yoe(fresh)
        role_tagged, off_archetype_count = role_filter.tag_by_role(track, yoe_kept)
        filtered_by_track[track] = role_tagged
        _log(
            log,
            f"  {track}: {len(with_apply)} apply ({dist_str}) · "
            f"{len(fresh)} unseen · "
            f"{len(yoe_kept)} under-YoE-cap (-{yoe_dropped}) · "
            f"{len(role_tagged)} retained "
            f"({off_archetype_count} off-archetype tagged, not dropped)",
        )

    # 3. Match — Voyage embedding similarity per track
    scored_by_track: dict[str, list[dict[str, Any]]] = {}
    for track_prefs in prefs.get("tracks", []):
        track = track_prefs["track"]
        if track not in filtered_by_track:
            continue
        posts = filtered_by_track[track]
        if not posts:
            scored_by_track[track] = []
            continue
        try:
            scored = matcher.score_posts_for_track(track_prefs["resume_summary"], posts)
            top = matcher.filter_and_rerank_target(scored)
            scored_by_track[track] = top
            # Archetype split of the top-N headed into Haiku. This number is
            # the single best early-warning indicator for the 2026-04-24
            # growth_marketing zero-match pattern: if `pass` drops to zero,
            # Haiku only sees off-archetype posts and the track will
            # return 0 matches no matter how well tuned the threshold is.
            n_pass = sum(
                1 for p in top if p.get("role_filter_verdict") != "off_archetype"
            )
            n_off = len(top) - n_pass
            _log(
                log,
                f"  {track}: {len(scored)} scored · {len(top)} above similarity "
                f"{config.SIMILARITY_THRESHOLD} "
                f"(archetype split: {n_pass} pass, {n_off} off-archetype)",
            )
        except Exception as exc:
            _log(log, f"  {track}: match failed: {exc}")
            scored_by_track[track] = []

    # 4. Rerank with Claude Haiku — NO threshold cut at this stage.
    #
    # We deliberately pass score_threshold=0.0 so EVERY candidate the
    # reranker saw gets returned with its Haiku score attached. The real
    # threshold cut happens in stage 5 AFTER priority boosts are applied,
    # which lets a LinkedIn-post-from-a-Tier-1-remote-company with a
    # borderline Haiku 6.0 stack its +linkedin_post +tier1 +remote boost
    # (~+2.1) and clear the real 6.5 bar at a final 8.1. Previously the
    # threshold ran inside rerank(), so those boosted-eligible posts were
    # killed before the boost could fire — exactly the symptom the user
    # reported as "no LinkedIn posts in the final digest".
    reranked_by_track: dict[str, list[dict[str, Any]]] = {}
    for track_prefs in prefs.get("tracks", []):
        track = track_prefs["track"]
        posts = scored_by_track.get(track, [])
        if not posts:
            reranked_by_track[track] = []
            continue
        try:
            scored = reranker.rerank(
                track, track_prefs["resume_summary"], posts, score_threshold=0.0
            )
            reranked_by_track[track] = scored
            _log(log, f"  {track}: {len(scored)} scored by Haiku (pre-boost)")
        except Exception as exc:
            _log(log, f"  {track}: rerank failed: {exc}")
            reranked_by_track[track] = []

    # 5. Priority boosts: source (linkedin_post > job boards), company (Indian
    #    tier-1), remote, user city, apply-link-in-comments, hiring-intent,
    #    recruiter-style body language. Score is clamped to [0,10]; posts
    #    carry `priority_reasons` so the email can render transparent badges.
    #
    # The threshold cut happens HERE — after boosts are applied — so that a
    # post whose final (boosted) score crosses LLM_SCORE_THRESHOLD passes
    # even if its base Haiku score alone did not.
    boosted_by_track: dict[str, list[dict[str, Any]]] = {}
    for track, posts in reranked_by_track.items():
        if not posts:
            boosted_by_track[track] = []
            continue
        boosted = priority_ranker.apply_priority_boosts(posts)
        final = [
            p for p in boosted
            if p.get("llm_score", 0.0) >= config.LLM_SCORE_THRESHOLD
        ]
        n_boosted_kept = sum(1 for p in final if p.get("priority_boost", 0) > 0)
        n_lifted_by_boost = sum(
            1 for p in final
            if p.get("base_llm_score", 0.0) < config.LLM_SCORE_THRESHOLD
            and p.get("llm_score", 0.0) >= config.LLM_SCORE_THRESHOLD
        )
        boosted_by_track[track] = final
        _log(
            log,
            f"  {track}: {len(final)} above threshold {config.LLM_SCORE_THRESHOLD} "
            f"({n_boosted_kept} boosted, {n_lifted_by_boost} lifted over line by boost)",
        )

    # 5b. Per-track (company, title) dedup — collapses the
    #     "5× DevcommX GTM Engineer" pattern where the same role is posted
    #     multiple times by resharers or syndicated across sources under
    #     the same track. Runs AFTER boost so the keeper is chosen by
    #     FINAL score (a LinkedIn post with a Tier-1 Indian boost should
    #     beat a Naukri listing of the same role).
    if config.PER_TRACK_COMPANY_CAP > 0:
        before = sum(len(v) for v in boosted_by_track.values())
        boosted_by_track, cap_dropped = _per_track_company_title_cap(
            boosted_by_track, config.PER_TRACK_COMPANY_CAP
        )
        after = sum(len(v) for v in boosted_by_track.values())
        if cap_dropped:
            _log(
                log,
                f"Per-track company/title cap dropped {cap_dropped} post(s) "
                f"({before} → {after}).",
            )

    # 6. Cross-track dedup: a post that appeared in multiple tracks → assign best-fit
    final = _assign_unique_track(boosted_by_track, prefs, log)

    # 6b. Content-hash dedup: the SAME JOB can arrive with DIFFERENT URLs
    #     (same role syndicated to LinkedIn posts + LinkedIn jobs + Naukri).
    #     URL-key dedup at stage 6 doesn't catch those — the URLs differ.
    #     Fuzzy content-key collapse (company, title_normalized, location)
    #     does, keeping the highest-scored instance.
    #
    #     Flag-gated (config.ENABLE_CONTENT_DEDUP) so anyone debugging
    #     missing-match issues can disable in one step.
    if config.ENABLE_CONTENT_DEDUP:
        before = sum(len(v) for v in final.values())
        final = _content_dedup_across_tracks(final)
        after = sum(len(v) for v in final.values())
        if before != after:
            _log(log, f"Content-dedup collapsed {before - after} near-duplicate post(s).")

    # 7. Record seen (only when not dry run) — delivered_at timestamp drives
    #    the 3-day TTL window. Also prune any truly-stale entries so the seen
    #    file doesn't grow without bound.
    if not dry_run:
        all_kept = [p for ps in final.values() for p in ps]
        tracking.record_seen(all_kept)
        pruned = tracking.prune_seen()
        _log(log, f"Recorded {len(all_kept)} posts as seen · pruned {pruned} stale entries.")
    else:
        _log(log, "[dry-run] skipping seen-state update and email delivery.")

    _render_summary(final)

    # 8. Deliver (CSV + email)
    delivery_info: dict[str, Any] = {"sent": False, "csv_path": None}
    if not dry_run:
        delivery_info = delivery.deliver(final)
        _log(log, f"Delivery: sent={delivery_info['sent']} · csv={delivery_info['csv_path']}")
    else:
        csv_path = config.RESULTS_DIR / f"dryrun_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        delivery.write_csv(final, csv_path)
        delivery_info["csv_path"] = str(csv_path)
        _log(log, f"[dry-run] wrote preview CSV to {csv_path}")

    _save_log(log)
    return {"matches": final, "log": log, "delivery": delivery_info}


# ------------------------------ Per-track cap ------------------------------
#
# Within a single track, the same role is often posted multiple times (same
# company + same title, different URLs because of:
#   • The hiring team + 2-3 ICs all reshare the original hiring post
#   • LinkedIn Jobs + Naukri syndicate the same listing
#   • Different recruiters at the same company post the same opening
#
# The cross-track content-dedup stage only catches cross-track duplicates
# (the same role appearing in BOTH growth_marketing AND strategy_consulting).
# This cap runs BEFORE that stage and catches same-track duplicates, keeping
# the top-N (default 1) by final boosted score. Source tie-break mirrors the
# cross-track dedup: linkedin_post > linkedin_job > naukri.


def _per_track_company_title_cap(
    boosted_by_track: dict[str, list[dict[str, Any]]],
    cap: int,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """For each track, collapse posts with the same (company, title) signature.

    Uses the same normalization as the cross-track dedup
    (`_normalize_company_for_dedup` / `_normalize_title_for_dedup`) so the
    two stages see identical clusters. Posts with empty company AND empty
    title are left untouched (key too broad to trust).

    Keeps the top `cap` posts per cluster ranked by:
        (llm_score desc, source_preference desc)

    Returns (deduped_by_track, total_dropped_count).
    """
    if cap <= 0:
        return boosted_by_track, 0

    out: dict[str, list[dict[str, Any]]] = {}
    dropped_total = 0
    for track, posts in boosted_by_track.items():
        clusters: dict[tuple[str, str], list[dict[str, Any]]] = {}
        untouched: list[dict[str, Any]] = []
        for p in posts:
            # Prefer structured fields; fall back to author-headline parsing
            # (priority_ranker._extract_author_company) when the source
            # doesn't populate `company` directly.
            raw_company = (
                p.get("company")
                or p.get("company_name")
                or priority_ranker._extract_author_company(p)
                or ""
            )
            company = _normalize_company_for_dedup(raw_company)
            title = _normalize_title_for_dedup(p.get("title") or "")
            if not company and not title:
                untouched.append(p)
                continue
            clusters.setdefault((company, title), []).append(p)

        kept: list[dict[str, Any]] = list(untouched)
        for entries in clusters.values():
            if len(entries) <= cap:
                kept.extend(entries)
                continue
            # Highest score wins; source preference is the tiebreaker (same
            # ordering as the cross-track content-dedup stage).
            entries.sort(
                key=lambda e: (
                    float(e.get("llm_score", 0.0)),
                    _SOURCE_PREFERENCE.get(e.get("source") or "", 0),
                ),
                reverse=True,
            )
            kept.extend(entries[:cap])
            dropped_total += len(entries) - cap

        kept.sort(key=lambda p: p.get("llm_score", 0.0), reverse=True)
        out[track] = kept
    return out, dropped_total


def _assign_unique_track(
    reranked_by_track: dict[str, list[dict[str, Any]]],
    prefs: dict[str, Any],
    log: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """If a post appears in multiple tracks, keep it only under its best-fit track."""
    track_defs = {t["track"]: t for t in prefs.get("tracks", [])}
    key_to_tracks: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for track, posts in reranked_by_track.items():
        for p in posts:
            k = tracking.post_key(p)
            key_to_tracks.setdefault(k, []).append((track, p))

    winners: dict[str, tuple[str, dict[str, Any]]] = {}
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    for k, entries in key_to_tracks.items():
        if len(entries) == 1:
            winners[k] = entries[0]
            continue
        # Multiple tracks competed. Ask Haiku to pick the best-fit track from the candidates.
        candidate_tracks = [{"track": t, **track_defs.get(t, {})} for t, _ in entries]
        first_post = entries[0][1]
        try:
            best = reranker.assign_best_track(first_post, candidate_tracks, client=client)
        except Exception as exc:
            _log(log, f"  best-track assignment failed: {exc}; defaulting to highest score")
            best = max(entries, key=lambda e: e[1].get("llm_score", 0.0))[0]
        match = next((e for e in entries if e[0] == best), entries[0])
        winners[k] = match

    out: dict[str, list[dict[str, Any]]] = {t: [] for t in reranked_by_track.keys()}
    for track, post in winners.values():
        out.setdefault(track, []).append(post)
    for track in out:
        out[track].sort(key=lambda p: p.get("llm_score", 0.0), reverse=True)
    return out


# ----------------------------- Content-hash dedup -----------------------------
#
# The same role gets syndicated under DIFFERENT URLs (same job on LinkedIn
# posts + LinkedIn jobs + Naukri), so stage 6's URL-key dedup can't catch it.
# This stage collapses those by a fuzzy content key: (company, title_head,
# location_head).
#
# Precision-first:
#   - Company is normalized by stripping common legal suffixes ("Pvt Ltd",
#     "Inc", "LLC", …) so "Acme Corp" ≡ "Acme Inc" ≡ "Acme".
#   - Title is lowercased, punctuation stripped, seniority modifiers
#     dropped (senior / lead / staff / …), then the first 4 meaningful
#     tokens form the head. "Senior Product Manager" and "Lead Product
#     Manager" both collapse to "product manager".
#   - Location takes the first comma-separated token. Bangalore/Bengaluru
#     and Gurgaon/Gurugram aliases are normalized.
#   - A post with BLANK company AND BLANK title skips dedup entirely — the
#     key would be ("", "", location), way too broad to trust.
#
# Winner-selection:
#   - Highest llm_score in each cluster.
#   - Ties broken by source preference: linkedin_post > linkedin_job >
#     naukri (mirrors the user's stated preference for LinkedIn posts).

_COMPANY_SUFFIX_RE = re.compile(
    r"\s+(?:pvt\.?|private|ltd\.?|limited|inc\.?|llc|llp|corporation|corp\.?|company|co\.?|"
    r"technologies|tech|solutions|services|industries|group|holdings)\s*$",
    re.I,
)

_TITLE_PUNCT_RE = re.compile(r"[^\w\s]")

# Seniority / structural tokens that shouldn't define the role signature.
# Removing them lets "Senior PM" and "Lead PM" collapse. Conservative — we
# do NOT drop skill/team tokens ("growth", "retention", "platform") because
# those DO change which role is being hired for.
_TITLE_STOPWORDS = {
    "senior", "sr", "lead", "staff", "principal", "junior", "jr",
    "associate", "assistant", "chief", "head", "vp", "director",
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "at",
}

_LOCATION_ALIASES = {
    "bengaluru": "bangalore",
    "gurugram": "gurgaon",
}

# Source preference (higher = preferred) — tiebreaker only; llm_score wins first.
_SOURCE_PREFERENCE = {"linkedin_post": 3, "linkedin_job": 2, "naukri": 1}


def _normalize_company_for_dedup(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip().lower()
    # Loop-strip legal suffixes — "Acme Technologies Pvt Ltd" needs two passes.
    prev: str | None = None
    while prev != s:
        prev = s
        s = _COMPANY_SUFFIX_RE.sub("", s).strip()
    return s


def _normalize_title_for_dedup(raw: str) -> str:
    if not raw:
        return ""
    s = _TITLE_PUNCT_RE.sub(" ", raw.lower())
    tokens = [t for t in s.split() if t and t not in _TITLE_STOPWORDS]
    # First 4 meaningful tokens — enough to disambiguate role families
    # without demanding a full exact-title match.
    return " ".join(tokens[:4])


def _normalize_location_for_dedup(raw: str) -> str:
    if not raw:
        return ""
    head = raw.split(",", 1)[0].strip().lower()
    return _LOCATION_ALIASES.get(head, head)


def _content_key(post: dict[str, Any]) -> tuple[str, str, str] | None:
    """Compute the fuzzy content key for dedup; None → 'skip this post'."""
    raw_company = (post.get("company") or post.get("company_name") or "").strip()
    if not raw_company:
        # Last-resort: parse the author's headline ("Head of Growth at Acme").
        extracted = priority_ranker._extract_author_company(post)
        if extracted:
            raw_company = extracted
    company = _normalize_company_for_dedup(raw_company)
    title = _normalize_title_for_dedup(post.get("title") or "")
    location = _normalize_location_for_dedup(post.get("location") or "")

    # Empty company AND empty title → key too broad, skip dedup. A post
    # with only a location ("", "", "bangalore") would wrongly collapse
    # every remote/Bangalore match into one winner.
    if not company and not title:
        return None
    return (company, title, location)


def _content_dedup_across_tracks(
    matches_by_track: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Collapse near-duplicate posts across tracks via fuzzy content key.

    Posts with no computable key fall through untouched (the URL-key dedup
    at stage 6 has already removed exact duplicates). Within each content
    cluster we keep the highest-llm_score instance; ties go to the source
    with the richer recruiter signal (linkedin_post > linkedin_job > naukri).

    Output shape mirrors the input — each kept post stays in its original
    track. Posts are re-sorted by score within each track for consistency
    with stage-6's output.
    """
    clusters: dict[tuple[str, str, str], list[tuple[str, dict[str, Any]]]] = {}
    untouched: list[tuple[str, dict[str, Any]]] = []
    for track, posts in matches_by_track.items():
        for p in posts:
            key = _content_key(p)
            if key is None:
                untouched.append((track, p))
            else:
                clusters.setdefault(key, []).append((track, p))

    winners: list[tuple[str, dict[str, Any]]] = list(untouched)
    for entries in clusters.values():
        if len(entries) == 1:
            winners.append(entries[0])
            continue
        # Highest score wins; source preference breaks ties.
        entries.sort(
            key=lambda e: (
                float(e[1].get("llm_score", 0.0)),
                _SOURCE_PREFERENCE.get(e[1].get("source") or "", 0),
            ),
            reverse=True,
        )
        winners.append(entries[0])

    out: dict[str, list[dict[str, Any]]] = {t: [] for t in matches_by_track.keys()}
    for track, post in winners:
        out.setdefault(track, []).append(post)
    for track in out:
        out[track].sort(key=lambda p: p.get("llm_score", 0.0), reverse=True)
    return out


def _render_summary(matches_by_track: dict[str, list[dict[str, Any]]]) -> None:
    tbl = Table(title="Match summary", border_style="green")
    tbl.add_column("Track", style="cyan")
    tbl.add_column("Matches", justify="right")
    tbl.add_column("Top score", justify="right")
    for t, ps in matches_by_track.items():
        top = max((p.get("llm_score", 0.0) for p in ps), default=0.0)
        tbl.add_row(t, str(len(ps)), f"{top:.1f}")
    console.print(tbl)


def _save_log(log: list[str]) -> None:
    path = config.RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(log), encoding="utf-8")


def run_setup() -> None:
    """Interactive resume upload + role selection for every resume PDF in RESUME_DIR."""
    from .resume_analyzer import (
        analyze_resume,
        prefer_longer_variant,
        save_profile,
        track_name_from_filename,
    )

    config.ensure_dirs()
    resume_dir = Path(config.RESUME_DIR)
    if not resume_dir.exists():
        console.print(f"[red]Resume dir {resume_dir} not found.[/]")
        return

    pdfs = sorted(resume_dir.glob("*.pdf"))
    if not pdfs:
        console.print(f"[red]No PDF resumes in {resume_dir}.[/]")
        return

    pdfs = prefer_longer_variant(pdfs)
    console.print(f"[cyan]Found {len(pdfs)} resume(s).[/]")
    for p in pdfs:
        console.print(f"  · {p.name}  →  track [bold]{track_name_from_filename(p)}[/]")

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    tracks: list[dict[str, Any]] = []
    for pdf in pdfs:
        track = track_name_from_filename(pdf)
        console.print(f"\n[bold]Analyzing[/] {pdf.name} …")
        profile = analyze_resume(pdf, client=client)
        save_profile(track, profile)
        prefs = role_selector.confirm_roles_for_track(track, profile)
        tracks.append(prefs)

    path = role_selector.save_preferences(tracks)
    console.print(f"\n[green]Saved preferences to {path}[/]")


def run_status() -> None:
    """Print seen/applied/rejected counts + last run summary."""
    stats = tracking.stats()
    console.print(f"[cyan]Tracking stats:[/] {json.dumps(stats)}")
    try:
        prefs = role_selector.load_preferences()
        tbl = Table(title="Configured tracks", border_style="cyan")
        tbl.add_column("Track")
        tbl.add_column("Active")
        tbl.add_column("Roles", overflow="fold")
        tbl.add_column("Cities")
        for t in prefs.get("tracks", []):
            tbl.add_row(
                t["track"],
                "✓" if t.get("active", True) else "✗",
                ", ".join(t.get("roles", [])),
                ", ".join(t.get("cities", [])),
            )
        console.print(tbl)
    except FileNotFoundError:
        console.print("[yellow]No preferences yet. Run `python -m src setup`.[/]")

    latest_logs = sorted(config.RESULTS_DIR.glob("run_*.log"))[-3:]
    for p in latest_logs:
        console.print(f"\n[dim]{p.name}[/]")
        console.print(p.read_text(encoding="utf-8")[-1200:])


def _print_exc() -> None:
    console.print(f"[red]{traceback.format_exc()}[/]")
