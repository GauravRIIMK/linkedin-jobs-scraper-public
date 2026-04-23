"""End-to-end diagnostic: probe every API's health, then run a verbose pass
through the pipeline per track printing every intermediate result so we can
see *why* the reranker is dropping posts (or any other silent failure).

Entry point: `python -m src diagnose`
"""
from __future__ import annotations

import json
import time
from typing import Any

from anthropic import Anthropic
from apify_client import ApifyClient
from rich.console import Console
from rich.table import Table

from . import (
    apify_runner,
    application_filter,
    config,
    jd_extractor,
    matcher,
    query_builder,
    reranker,
    role_selector,
)

console = Console()


# ---------- API HEALTH PROBES ----------

def _probe_apify() -> dict[str, Any]:
    """Cheapest Apify call: fetch authenticated user info. No actor run → no cost."""
    t0 = time.time()
    try:
        client = ApifyClient(config.APIFY_TOKEN)
        info = client.user("me").get() or {}
        return {
            "ok": True,
            "ms": int((time.time() - t0) * 1000),
            "user": info.get("username") or info.get("id") or "unknown",
            "plan": info.get("plan") or "?",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "err": f"{type(exc).__name__}: {exc}"[:220]}


def _probe_voyage() -> dict[str, Any]:
    """1-token embedding call."""
    t0 = time.time()
    try:
        import voyageai

        client = voyageai.Client(api_key=config.VOYAGE_API_KEY)
        r = client.embed(["diagnostic"], model=config.VOYAGE_MODEL, input_type="document")
        return {
            "ok": True,
            "ms": int((time.time() - t0) * 1000),
            "model": config.VOYAGE_MODEL,
            "dim": len(r.embeddings[0]),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "model": config.VOYAGE_MODEL, "err": f"{type(exc).__name__}: {exc}"[:220]}


def _probe_anthropic() -> dict[str, Any]:
    """1-token Haiku call."""
    t0 = time.time()
    try:
        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        r = client.messages.create(
            model=config.CLAUDE_MODEL_LIGHT,
            max_tokens=8,
            messages=[{"role": "user", "content": "Say OK."}],
        )
        text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        return {
            "ok": True,
            "ms": int((time.time() - t0) * 1000),
            "model": config.CLAUDE_MODEL_LIGHT,
            "reply": text.strip()[:40],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "model": config.CLAUDE_MODEL_LIGHT, "err": f"{type(exc).__name__}: {exc}"[:220]}


def _probe_resend() -> dict[str, Any]:
    """Validate the send-only Resend API key by POSTing an intentionally-empty
    payload. A send-only key returns 401 on GET /api-keys (that endpoint needs
    a full-access key) but 422 on POST /emails with missing fields — which
    proves the key is authenticated. 401 is the real failure mode."""
    import requests

    t0 = time.time()
    try:
        r = requests.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {config.RESEND_API_KEY}",
                "Content-Type": "application/json",
            },
            json={},  # empty body → expect 422 (auth'd) vs 401 (bad key)
            timeout=10,
        )
        ms = int((time.time() - t0) * 1000)
        # 200/202 = unlikely with empty payload but fine; 422 = valid key, bad payload;
        # 400 = malformed request (key authed); 401/403 = auth failure.
        authed = r.status_code not in (401, 403)
        return {
            "ok": authed,
            "ms": ms,
            "status": r.status_code,
            "note": "422 expected w/ empty body" if r.status_code == 422 else "",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "err": f"{type(exc).__name__}: {exc}"[:220]}


def render_health(results: dict[str, dict[str, Any]]) -> None:
    t = Table(title="API health", border_style="green")
    t.add_column("Service", style="cyan")
    t.add_column("OK", justify="center")
    t.add_column("Latency", justify="right")
    t.add_column("Detail", overflow="fold")
    for k, v in results.items():
        ok = "[green]✓[/]" if v.get("ok") else "[red]✗[/]"
        lat = f"{v.get('ms', '-')} ms" if "ms" in v else "-"
        if v.get("err"):
            detail = f"[red]{v['err']}[/]"
        else:
            keep = {x: v[x] for x in v if x not in ("ok", "ms", "err")}
            detail = json.dumps(keep)
        t.add_row(k, ok, lat, detail)
    console.print(t)


# ---------- PER-TRACK VERBOSE SCAN ----------

def verbose_scan_for_track(
    track_prefs: dict[str, Any],
    *,
    queries_cap: int = 3,
    posts_cap: int = 15,
) -> dict[str, Any]:
    """Run the pipeline for a single track, logging every intermediate result.

    Caps scraping cost by limiting queries + posts-per-query.
    Returns a summary dict so the caller can render a cross-track table.
    """
    track = track_prefs["track"]
    console.print(f"\n[bold cyan]━━━ {track} ━━━[/]")

    # 1. Queries
    queries = query_builder.build_queries_for_track(track_prefs)
    queries = queries[:queries_cap]
    console.print(f"[dim]{len(queries)} queries (capped):[/] {queries}")

    # 2. Scrape
    client = ApifyClient(config.APIFY_TOKEN)
    raw: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for q in queries:
        try:
            got = apify_runner.run_query(client, q, max_posts=posts_cap)
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]query {q!r} failed:[/] {exc}")
            continue
        before = len(raw)
        for p in got:
            url = p.get("post_url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                raw.append(p)
        console.print(f"  · {q!r} → {len(got)} posts ({len(raw) - before} new)")
    console.print(f"[cyan]raw (deduped):[/] {len(raw)}")

    if not raw:
        return {
            "track": track,
            "raw": 0,
            "with_apply": 0,
            "apply_dist": {},
            "above_sim": 0,
            "llm_kept": 0,
            "top_llm": 0.0,
        }

    # 3. Application-method filter (gform | ATS | Easy Apply | mail)
    with_apply = application_filter.filter_posts_with_apply_method(raw)
    dist = application_filter.method_distribution(with_apply)
    dist_str = ", ".join(f"{k}={v}" for k, v in sorted(dist.items())) or "none"
    console.print(f"[cyan]with apply-method:[/] {len(with_apply)} / {len(raw)} · {dist_str}")
    for i, p in enumerate(with_apply[:5]):
        methods = ",".join(p.get("apply_methods") or [])
        console.print(
            f"  {i+1}. [{methods:<22}] "
            f"{(p.get('author_name') or '?')[:22]:<22} · "
            f"{(p.get('author_title') or '')[:36]:<36} · "
            f"{(p.get('post_url') or '')[-40:]}"
        )

    if not with_apply:
        return {
            "track": track,
            "raw": len(raw),
            "with_apply": 0,
            "apply_dist": {},
            "above_sim": 0,
            "llm_kept": 0,
            "top_llm": 0.0,
        }

    # 4. JD extraction
    enriched = [{**p, **jd_extractor.extract_jd(p)} for p in with_apply]
    jd_len = [len(p.get("jd_text") or "") for p in enriched]
    console.print(
        f"[cyan]JD extraction:[/] avg={sum(jd_len) // len(jd_len)} chars, "
        f"min={min(jd_len)}, max={max(jd_len)}"
    )

    # 5. Similarity (Voyage)
    try:
        scored = matcher.score_posts_for_track(track_prefs["resume_summary"], enriched)
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]Voyage scoring failed:[/] {exc}")
        return {
            "track": track,
            "raw": len(raw),
            "with_apply": len(with_apply),
            "apply_dist": dist,
            "above_sim": 0,
            "llm_kept": 0,
            "top_llm": 0.0,
        }

    console.print(f"[cyan]similarity (threshold {config.SIMILARITY_THRESHOLD}):[/]")
    above_sim = 0
    for p in scored:
        ok = p["similarity"] >= config.SIMILARITY_THRESHOLD
        above_sim += 1 if ok else 0
        mark = "[green]✓[/]" if ok else "[dim]✗[/]"
        console.print(
            f"  {mark} sim={p['similarity']:.3f} · "
            f"{(p.get('author_title') or p.get('text', '')[:55])[:55]}"
        )

    candidates = matcher.filter_and_rerank_target(scored)
    console.print(f"[cyan]→ sent to Haiku rerank:[/] {len(candidates)}")
    if not candidates:
        return {
            "track": track,
            "raw": len(raw),
            "with_apply": len(with_apply),
            "apply_dist": dist,
            "above_sim": above_sim,
            "llm_kept": 0,
            "top_llm": 0.0,
        }

    # 6. Haiku rerank — log EVERY score + reason so we can see the truth
    anth = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    llm_scores: list[float] = []
    llm_kept = 0
    console.print(f"[cyan]Haiku rerank (threshold {config.LLM_SCORE_THRESHOLD}):[/]")
    for p in candidates:
        try:
            r = reranker.score_one(track, track_prefs["resume_summary"], p, client=anth)
        except Exception as exc:  # noqa: BLE001
            console.print(f"  [red]Haiku call failed:[/] {exc}")
            continue
        score = float(r.get("score", 0.0))
        reason = (r.get("reason") or "").strip()
        llm_scores.append(score)
        kept = score >= config.LLM_SCORE_THRESHOLD
        llm_kept += 1 if kept else 0
        mark = "[green]✓[/]" if kept else "[yellow]✗[/]"
        console.print(
            f"  {mark} score={score:.1f}/{config.LLM_SCORE_THRESHOLD} · {reason[:110]}"
        )
        console.print(f"     [dim]url: {(p.get('post_url') or '')[-75:]}[/]")

    return {
        "track": track,
        "raw": len(raw),
        "with_apply": len(with_apply),
        "apply_dist": dist,
        "above_sim": above_sim,
        "llm_kept": llm_kept,
        "top_llm": max(llm_scores) if llm_scores else 0.0,
    }


def _render_summary(rows: list[dict[str, Any]]) -> None:
    t = Table(title="Diagnostic summary", border_style="cyan")
    t.add_column("Track")
    t.add_column("Raw", justify="right")
    t.add_column("W/ apply", justify="right")
    t.add_column("Methods", overflow="fold")
    t.add_column("Above sim", justify="right")
    t.add_column("Haiku kept", justify="right")
    t.add_column("Top LLM", justify="right")
    for r in rows:
        dist = r.get("apply_dist") or {}
        dist_str = ", ".join(f"{k}={v}" for k, v in sorted(dist.items())) or "-"
        t.add_row(
            r["track"],
            str(r["raw"]),
            str(r.get("with_apply", 0)),
            dist_str,
            str(r["above_sim"]),
            str(r["llm_kept"]),
            f"{r['top_llm']:.1f}",
        )
    console.print(t)


def main() -> None:
    console.print("[bold cyan]══ API HEALTH ══[/]")
    results = {
        "apify": _probe_apify(),
        "voyage": _probe_voyage(),
        "anthropic": _probe_anthropic(),
        "resend": _probe_resend(),
    }
    render_health(results)

    if not all(v.get("ok") for v in results.values()):
        console.print(
            "\n[yellow]⚠ One or more APIs are unhealthy. "
            "Running pipeline diagnostics anyway — downstream steps may fail.[/]"
        )

    console.print("\n[bold cyan]══ PIPELINE (all active tracks) ══[/]")
    prefs = role_selector.load_preferences()
    rows: list[dict[str, Any]] = []
    for tp in prefs.get("tracks", []):
        if not tp.get("active", True):
            continue
        try:
            rows.append(verbose_scan_for_track(tp))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]{tp['track']} diagnostic crashed:[/] {exc}")
            rows.append({
                "track": tp["track"], "raw": 0, "with_apply": 0, "apply_dist": {},
                "above_sim": 0, "llm_kept": 0, "top_llm": 0.0,
            })

    console.print("\n[bold cyan]══ SUMMARY ══[/]")
    _render_summary(rows)


if __name__ == "__main__":
    main()
