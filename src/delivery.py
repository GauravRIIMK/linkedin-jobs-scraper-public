"""Email digest + CSV writer for daily matches.

Two delivery backends, selectable via `DELIVERY_METHOD` env var:

- **`resend`** (recommended, default) — uses the Resend HTTP API with an API key.
  No Gmail password required. Rotatable from https://resend.com/api-keys.
  Free tier: 100 emails/day, 3,000/month. Emails still arrive in your Gmail inbox
  because the `To:` is your own address.

- **`gmail`** — SMTP over SSL with a Gmail app password. Works but requires 2FA +
  a 16-char app password in `GMAIL_APP_PASSWORD`.

Layout decisions:
  - Subject line includes today's date (not "week of X") since the cron is daily.
  - Body header says "Your daily LinkedIn matches" (was incorrectly "Weekly").
  - Each match card renders COLORED BADGES for priority_reasons (Tier-1
    Indian company, Remote, Apply in comments, etc.) so the user sees WHY
    a match was boosted without digging into `llm_reason`.
  - Cards sort by boosted llm_score desc within each track (already done
    upstream by priority_ranker).
"""
from __future__ import annotations

import base64
import csv
import smtplib
import time
from datetime import date
from email.message import EmailMessage
from html import escape
from pathlib import Path
from typing import Any

import requests

from . import config

CSV_FIELDS = [
    "track",
    "post_url",
    "author_name",
    "author_title",
    "posted_at",
    "llm_score",
    "base_llm_score",
    "priority_boost",
    "priority_reasons",
    "similarity",
    "llm_reason",
    "apply_methods",
    "apply_urls",
    "apply_emails",
    "gform_urls",        # retained for backward compatibility with earlier runs
    "jd_doc_urls",
    "jd_snippet",
]


def _day_label() -> str:
    n = config.POSTED_LIMIT_DAYS
    if n <= 1:
        return "last 24 hours"
    return f"last {n} days"


def write_csv(matches_by_track: dict[str, list[dict[str, Any]]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for track, matches in matches_by_track.items():
            for m in matches:
                writer.writerow(
                    {
                        "track": track,
                        "post_url": m.get("post_url", ""),
                        "author_name": m.get("author_name", ""),
                        "author_title": m.get("author_title", ""),
                        "posted_at": m.get("posted_at", ""),
                        "llm_score": round(float(m.get("llm_score", 0.0)), 2),
                        "base_llm_score": round(float(m.get("base_llm_score", m.get("llm_score", 0.0))), 2),
                        "priority_boost": round(float(m.get("priority_boost", 0.0)), 2),
                        "priority_reasons": ", ".join(m.get("priority_reasons") or []),
                        "similarity": round(float(m.get("similarity", 0.0)), 3),
                        "llm_reason": m.get("llm_reason", ""),
                        "apply_methods": ", ".join(m.get("apply_methods") or []),
                        "apply_urls": " | ".join(m.get("apply_urls") or []),
                        "apply_emails": ", ".join(m.get("apply_emails") or []),
                        "gform_urls": " | ".join(m.get("gform_urls") or []),
                        "jd_doc_urls": " | ".join(m.get("jd_doc_urls") or []),
                        "jd_snippet": (m.get("jd_text") or "")[:280].replace("\n", " "),
                    }
                )
    return out_path


def build_subject(matches_by_track: dict[str, list[dict[str, Any]]]) -> str:
    parts = [f"{len(v)} {k.replace('_', ' ').title()}" for k, v in matches_by_track.items() if v]
    today = date.today().isoformat()
    if not parts:
        return f"LinkedIn digest · no matches · {today}"
    return f"LinkedIn digest · {' + '.join(parts)} · {today}"


def _method_label(method: str) -> str:
    """Short human label for an apply method key."""
    return {
        "gform": "Google Form",
        "lever": "Lever",
        "greenhouse": "Greenhouse",
        "ashby": "Ashby",
        "workday": "Workday",
        "workable": "Workable",
        "smartrecruiters": "SmartRecruiters",
        "recruitee": "Recruitee",
        "teamtailor": "Teamtailor",
        "jobvite": "Jobvite",
        "icims": "iCIMS",
        "bamboohr": "BambooHR",
        "keka": "Keka",
        "zoho": "Zoho",
        "wellfound": "Wellfound",
        "breezy": "Breezy",
        "dover": "Dover",
        "naukri": "Naukri",
        "instahyre": "Instahyre",
        "cutshort": "Cutshort",
        "linkedin_easy_apply": "LinkedIn Apply",
        "mail": "Mail",
        "soft_dm": "DM / comment",
        "linkedin_job": "LinkedIn Jobs",
    }.get(method, method.title())


def _render_apply_links(m: dict[str, Any]) -> str:
    """Build the HTML fragment of clickable apply links for one match.

    Shows one labeled link per URL, mailto: links for any apply_emails, and
    falls back to the legacy gform_urls list when the post was produced by
    an older pipeline version that didn't populate apply_urls.
    """
    chunks: list[str] = []
    urls = m.get("apply_urls") or []
    # If apply_urls is empty (legacy record), fall back to gform_urls
    if not urls:
        for i, u in enumerate(m.get("gform_urls") or []):
            chunks.append(f'<a href="{escape(u)}">Form {i+1}</a>')
    else:
        for i, u in enumerate(urls):
            chunks.append(f'<a href="{escape(u)}">Apply {i+1}</a>')
    for e in m.get("apply_emails") or []:
        chunks.append(f'<a href="mailto:{escape(e)}">{escape(e)}</a>')
    if not chunks:
        return '<span style="color:#999">no direct link</span>'
    return " · ".join(chunks)


# Badge palette: (background, text_color) tuples keyed by priority reason.
# Stable color per reason so the user trains their eye to recognize them.
_BADGE_COLORS: dict[str, tuple[str, str]] = {
    "Tier-1 Indian company":     ("#fff4e5", "#b45309"),   # amber — "premium"
    "Remote":                    ("#e8f5e9", "#2e7d32"),   # green — "go signal"
    "Near you":                  ("#e3f2fd", "#0277bd"),   # blue — "local"
    "LinkedIn post":             ("#eaf1fb", "#0a66c2"),   # LinkedIn blue
    "Apply in comments":         ("#f3e8ff", "#6b21a8"),   # purple — "hand-picked"
    "Recruiter-dropped link":    ("#fce4ec", "#ad1457"),   # pink — "OP self-reply"
    "Hiring manager / recruiter":("#ffe7d1", "#c2410c"),   # orange — "direct hirer"
    "Recruiter-style post":      ("#fef3c7", "#92400e"),   # soft yellow — "body language"
    # Network-degree badges — eye-catchy greens the user learns to look for.
    # 1st uses a deeper green than "Remote" so the two don't blur together.
    "1st connection":            ("#dcfce7", "#14532d"),   # deep green — "direct network"
    "2nd connection":            ("#ecfccb", "#3f6212"),   # lime — "shared connection"
    # Geolocation badges — the user prioritizes India hires.
    "India":                     ("#fff1f2", "#9f1239"),   # soft rose — "country match"
    "Not in India":              ("#f1f5f9", "#475569"),   # muted gray — "deprioritized"
    # Archetype-mismatch warning chip. Red-ish so the user sees the trade-off
    # at a glance: "this post is off-archetype but other signals lifted it
    # past the threshold anyway".
    "Off-archetype":             ("#fef2f2", "#991b1b"),   # light red — "warning"
}


def _format_magnitude(mag: float) -> str:
    """Format a signed boost/penalty for display inside a chip.

    Convention: always show the sign (`+0.4`, `-1.2`) so users can
    distinguish boosts from penalties at a glance. Round to 1 decimal —
    tighter precision would just add visual noise without improving
    interpretability.
    """
    if mag > 0:
        return f" +{mag:.1f}"
    if mag < 0:
        return f" {mag:.1f}"
    return ""


def _render_badges(reasons: list) -> str:
    """Render a row of colored chips for the priority reasons.

    Accepts either:
      - list[str]                       — legacy shape, label-only chips
      - list[tuple[str, float]]         — detailed shape, chip shows
                                          `Label ±X.X` to expose the
                                          boost/penalty magnitude

    Both shapes are supported so the email renders correctly even if an
    older CSV is replayed (priority_reasons list[str]) or an upgraded
    priority_ranker emits detailed tuples.
    """
    if not reasons:
        return ""
    chips: list[str] = []
    for r in reasons:
        if isinstance(r, (tuple, list)) and len(r) == 2:
            label_raw, magnitude_raw = r[0], r[1]
            try:
                magnitude = float(magnitude_raw)
            except (TypeError, ValueError):
                magnitude = 0.0
            label = str(label_raw)
            mag_str = _format_magnitude(magnitude)
        else:
            label = str(r)
            mag_str = ""
        bg, fg = _BADGE_COLORS.get(label, ("#f4f4f4", "#555"))
        chips.append(
            f'<span style="background:{bg};color:{fg};padding:2px 7px;'
            f'border-radius:10px;font-size:11px;margin-right:4px;'
            f'white-space:nowrap;font-weight:600">'
            f'{escape(label)}{escape(mag_str)}'
            f'</span>'
        )
    return "".join(chips)


def build_html(matches_by_track: dict[str, list[dict[str, Any]]]) -> str:
    total_matches = sum(len(v) for v in matches_by_track.values())
    sections: list[str] = []
    for track, matches in matches_by_track.items():
        title = track.replace("_", " ").title()
        if not matches:
            sections.append(
                f'<h2 style="margin-top:24px;color:#0a66c2">{escape(title)}</h2>'
                f'<p style="color:#999;font-size:13px;margin-top:4px"><em>No matches today.</em></p>'
            )
            continue
        rows = []
        for m in matches:
            apply_links_html = _render_apply_links(m)
            doc_links = " · ".join(
                f'<a href="{escape(u)}">JD doc</a>' for u in (m.get("jd_doc_urls") or [])
            )
            reason = escape(m.get("llm_reason") or "")
            score = round(float(m.get("llm_score", 0.0)), 1)
            base_score = round(float(m.get("base_llm_score", score)), 1)
            boost = float(m.get("priority_boost", 0.0))
            post_url = m.get("post_url") or "#"
            author = escape(m.get("author_name") or "Unknown")
            author_title = escape(m.get("author_title") or "")
            snippet = escape((m.get("jd_text") or "")[:260]).replace("\n", " ")
            methods = m.get("apply_methods") or []
            methods_chip = (
                f'<span style="font-size:11px;color:#555;background:#f0f0f0;'
                f'padding:2px 7px;border-radius:10px;margin-left:4px">'
                f'{escape(", ".join(_method_label(x) for x in methods))}</span>'
                if methods else ""
            )
            # Prefer the detailed (label, magnitude) list — falls back to the
            # legacy label-only list for posts rendered from older CSVs.
            badges_html = _render_badges(
                m.get("priority_reasons_detailed") or m.get("priority_reasons") or []
            )
            # Score chip: show `base → final` when boost/penalty moved the
            # score by ≥0.1, so the user sees how the final number came
            # together (e.g. `5.8 → 7.0/10` makes it obvious the post
            # cleared the bar via boosts rather than raw Haiku fit).
            if abs(score - base_score) >= 0.1:
                score_inner = f"{base_score:.1f} → {score:.1f}/10"
            else:
                score_inner = f"{score}/10"
            score_tag = (
                f'<span style="background:#0a66c2;color:#fff;padding:2px 8px;'
                f'border-radius:3px;font-weight:600;font-size:12px">'
                f'{score_inner}'
                f'</span>'
            )
            rows.append(
                f"""
                <div style="border-left:3px solid #0a66c2;padding:12px 14px;margin:12px 0;background:#fafafa;border-radius:4px">
                  <div style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;font-size:12px;color:#666">
                    {score_tag} {badges_html}{methods_chip}
                  </div>
                  <div style="margin-top:6px;color:#555;font-size:12px">
                    {author}{(" · " + author_title) if author_title else ""}
                  </div>
                  <div style="margin-top:8px;font-size:13px">
                    <a href="{escape(post_url)}" style="font-weight:600;color:#0a66c2;text-decoration:none">Open post</a>
                    &nbsp;·&nbsp; {apply_links_html}{(' &nbsp;·&nbsp; ' + doc_links) if doc_links else ''}
                  </div>
                  <div style="margin-top:8px;color:#333;font-size:13px"><em>{reason}</em></div>
                  <div style="margin-top:6px;color:#555;font-size:12px;line-height:1.5">{snippet}&hellip;</div>
                </div>
                """
            )
        sections.append(
            f'<h2 style="margin-top:24px;color:#0a66c2">{escape(title)} '
            f'<span style="color:#999;font-weight:400;font-size:15px">· {len(matches)}</span></h2>'
            + "".join(rows)
        )

    today = date.today().isoformat()
    # Daily-digest framing: even though the scraper looks back POSTED_LIMIT_DAYS
    # (typically 7) to be tolerant of missed runs, this email is the daily
    # digest — so the header talks about "today's" matches, not a week window.
    summary_line = (
        f"{total_matches} match{'' if total_matches == 1 else 'es'} across "
        f"{len([v for v in matches_by_track.values() if v])} track(s) from today's scrape, "
        "ranked by fit + priority boosts."
    ) if total_matches else (
        "No new matches today. The scraper ran successfully — "
        "this just means nothing crossed today's fit threshold."
    )

    return f"""
    <html>
      <body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;max-width:720px;margin:0 auto;padding:20px;color:#222;line-height:1.4">
        <h1 style="border-bottom:2px solid #0a66c2;padding-bottom:10px;margin:0 0 8px 0;font-size:22px">
          Your daily LinkedIn matches
        </h1>
        <p style="color:#666;font-size:13px;margin:0 0 6px 0">
          {summary_line}
        </p>
        <p style="color:#999;font-size:11px;margin:0 0 18px 0">
          Badges:
          <span style="background:#dcfce7;color:#14532d;padding:1px 6px;border-radius:10px;font-size:10px;font-weight:600">1st connection</span>
          <span style="background:#ecfccb;color:#3f6212;padding:1px 6px;border-radius:10px;font-size:10px;font-weight:600">2nd connection</span>
          <span style="background:#fff4e5;color:#b45309;padding:1px 6px;border-radius:10px;font-size:10px;font-weight:600">Tier-1 Indian</span>
          <span style="background:#fff1f2;color:#9f1239;padding:1px 6px;border-radius:10px;font-size:10px;font-weight:600">India</span>
          <span style="background:#e8f5e9;color:#2e7d32;padding:1px 6px;border-radius:10px;font-size:10px;font-weight:600">Remote</span>
          <span style="background:#f3e8ff;color:#6b21a8;padding:1px 6px;border-radius:10px;font-size:10px;font-weight:600">Apply in comments</span>
          <span style="background:#eaf1fb;color:#0a66c2;padding:1px 6px;border-radius:10px;font-size:10px;font-weight:600">LinkedIn post</span>
          <span style="background:#e3f2fd;color:#0277bd;padding:1px 6px;border-radius:10px;font-size:10px;font-weight:600">Near you</span>
          <span style="background:#fef2f2;color:#991b1b;padding:1px 6px;border-radius:10px;font-size:10px;font-weight:600">Off-archetype</span>
          <span style="color:#999;font-weight:400">&nbsp;· Numbers (e.g. <code>India +0.4</code>) show each chip's contribution to the final score.</span>
        </p>
        {''.join(sections)}
        <hr style="border:none;border-top:1px solid #eee;margin:28px 0 12px 0">
        <p style="color:#999;font-size:11px;margin:0">
          Generated {today} · CSV with all fields attached · Reply
          <code style="background:#f4f4f4;padding:1px 4px;border-radius:2px">reject &lt;url&gt;</code>
          to suppress a post from future digests, or
          <code style="background:#f4f4f4;padding:1px 4px;border-radius:2px">apply &lt;url&gt;</code>
          after applying so it never shows up again.
        </p>
      </body>
    </html>
    """


# ---------- Resend (HTTP API, no password) ----------


def _send_via_resend(
    subject: str,
    html_body: str,
    to_address: str,
    attachment_path: Path | None,
) -> None:
    if not config.RESEND_API_KEY:
        raise RuntimeError(
            "RESEND_API_KEY not set. Get a free key at https://resend.com/api-keys "
            "and add to .env as RESEND_API_KEY=re_..."
        )

    payload: dict[str, Any] = {
        "from": config.RESEND_FROM,
        "to": [to_address],
        "subject": subject,
        "html": html_body,
    }
    if attachment_path and attachment_path.exists():
        payload["attachments"] = [
            {
                "filename": attachment_path.name,
                "content": base64.b64encode(attachment_path.read_bytes()).decode("ascii"),
            }
        ]

    url = "https://api.resend.com/emails"
    headers = {
        "Authorization": f"Bearer {config.RESEND_API_KEY}",
        "Content-Type": "application/json",
    }

    last_err: str | None = None
    for attempt in range(3):  # 3 attempts total with exponential backoff
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=20)
        except requests.RequestException as exc:
            last_err = f"network: {exc}"
        else:
            if resp.status_code < 300:
                return
            # Retry on 5xx / 429; fail fast on 4xx (bad key, bad payload)
            if resp.status_code in (408, 429) or 500 <= resp.status_code < 600:
                last_err = f"{resp.status_code}: {resp.text[:400]}"
            else:
                raise RuntimeError(f"Resend API error {resp.status_code}: {resp.text[:400]}")
        time.sleep(2 ** attempt)  # 1s, 2s, 4s
    raise RuntimeError(f"Resend send failed after 3 attempts: {last_err}")


# ---------- Gmail SMTP (fallback) ----------


def _send_via_gmail(
    subject: str,
    html_body: str,
    to_address: str,
    attachment_path: Path | None,
) -> None:
    if not config.GMAIL_ADDRESS or not config.GMAIL_APP_PASSWORD:
        raise RuntimeError(
            "GMAIL_ADDRESS / GMAIL_APP_PASSWORD not set. Generate an app password at "
            "https://myaccount.google.com/apppasswords and add to .env."
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.GMAIL_ADDRESS
    msg["To"] = to_address
    msg.set_content("Your mail client does not support HTML. Open on a device that does.")
    msg.add_alternative(html_body, subtype="html")

    if attachment_path and attachment_path.exists():
        msg.add_attachment(
            attachment_path.read_bytes(),
            maintype="text",
            subtype="csv",
            filename=attachment_path.name,
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(config.GMAIL_ADDRESS, config.GMAIL_APP_PASSWORD)
        smtp.send_message(msg)


# ---------- Public API ----------


def send_email(
    subject: str,
    html_body: str,
    attachment_path: Path | None = None,
    to_address: str | None = None,
) -> None:
    to_address = to_address or config.EMAIL_TO
    if not to_address:
        raise RuntimeError("No recipient set. Fill EMAIL_TO or GMAIL_ADDRESS in .env.")

    method = config.DELIVERY_METHOD
    if method == "resend":
        _send_via_resend(subject, html_body, to_address, attachment_path)
    elif method == "gmail":
        _send_via_gmail(subject, html_body, to_address, attachment_path)
    else:
        raise RuntimeError(f"Unknown DELIVERY_METHOD={method!r}. Use 'resend' or 'gmail'.")


def deliver(matches_by_track: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Write CSV, send HTML email. Returns {csv_path, subject, sent, method, error?}."""
    today = date.today().isoformat()
    csv_path = config.RESULTS_DIR / f"matches_{today}.csv"
    write_csv(matches_by_track, csv_path)
    subject = build_subject(matches_by_track)
    html = build_html(matches_by_track)
    sent = False
    error: str | None = None
    try:
        send_email(subject, html, attachment_path=csv_path)
        sent = True
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (config.RESULTS_DIR / f"delivery_error_{today}.txt").write_text(
            error + "\n", encoding="utf-8"
        )
    return {
        "csv_path": str(csv_path),
        "subject": subject,
        "sent": sent,
        "method": config.DELIVERY_METHOD,
        "error": error,
    }
