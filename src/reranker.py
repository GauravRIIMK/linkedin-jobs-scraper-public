"""Claude Haiku reranks top candidates with a structured score + 1-line reasoning."""
from __future__ import annotations

import json
from typing import Any

from anthropic import Anthropic

from . import config

SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "description": "Fit score from 0 (no fit) to 10 (perfect fit). Consider skill overlap, seniority, domain.",
            "minimum": 0,
            "maximum": 10,
        },
        "reason": {
            "type": "string",
            "description": "One sentence (<25 words) explaining the score.",
        },
        "best_track_vote": {
            "type": "string",
            "description": "If scoring this against a pool of tracks, which track name fits best. Leave blank if scoring for a single track.",
        },
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Track-specific guardrails injected into the rerank prompt. Centralized here
# so track semantics aren't smeared across multiple prompts.
#
# NOTE: the hard filters in experience_filter + role_filter already block the
# most obvious seniority / coder-role misfits upstream. The hints below catch
# the subtler cases that slip past the regex gates (e.g. an SDE JD titled
# "Automation Engineer" that doesn't match our negative keyword list).
# ---------------------------------------------------------------------------
_TRACK_HINTS: dict[str, str] = {
    "ai_ops_automation": (
        "CANDIDATE ARCHETYPE: AI operator / vibecoder — uses AI tools, agents, "
        "and no-code/low-code platforms to ship products and automate "
        "business workflows WITHOUT traditional software engineering. They do "
        "NOT write production code, do NOT build ML models, and do NOT have a "
        "CS background.\n"
        "Score HIGH (7-10): 'AI operator / builder / specialist / strategist', "
        "automation specialist or lead, chief of staff, operations manager "
        "shipping with AI, AI-first product manager, workflow/process automation, "
        "AI transformation / enablement roles.\n"
        "Score LOW (0-4) regardless of other signals: Software Engineer, SDE, "
        "ML Engineer, Data Scientist, MLOps, DevOps, Research Engineer, "
        "language-specific developer roles (Python/Java/React/etc). These "
        "require coding depth the candidate does NOT have."
    ),
    "growth_marketing": (
        "CANDIDATE ARCHETYPE: growth / lifecycle / product marketer — owns "
        "brand, content, lifecycle (email/push/SMS), retention, community, "
        "SEO/content, and product marketing (PMM) charters. Works across "
        "marketing, creative, and product — does NOT write production code, "
        "does NOT run paid-media buying as a primary charter, and is NOT a "
        "GTM/sales engineer.\n"
        "Score HIGH (7-10): Growth Marketing Manager, Brand Marketing / "
        "Brand Manager, Product Marketing Manager (PMM), Lifecycle / Retention "
        "/ CRM Marketing, Content Marketing, Community Manager, SEO / Content "
        "Strategist, Marketing Lead / Head of Marketing at an early-stage "
        "consumer / SaaS company.\n"
        "Score LOW (0-4) regardless of other signals: Growth Engineer, GTM "
        "Engineer, Revenue Engineer, Marketing Engineer, Performance / Digital "
        "Marketing roles that are pure paid-ads buying (Google Ads / Meta Ads "
        "specialist with no lifecycle/brand/content scope), Sales Development "
        "Representative (SDR/BDR), Account Executive, Sales Manager. These "
        "roles either require engineering depth the candidate does NOT have "
        "or sit in the sales org rather than marketing."
    ),
}


def _rerank_prompt(
    track_name: str,
    resume_summary: str,
    jd: str,
    author_title: str,
) -> str:
    hint = _TRACK_HINTS.get(track_name, "")
    hint_block = f"\n\nTRACK GUARDRAIL:\n{hint}\n" if hint else "\n"
    yoe_clause = (
        f"The candidate has ~{config.USER_YOE:.1f} years of experience. Roles "
        f"asking for significantly more than {config.USER_YOE + config.YOE_BUFFER:.0f} "
        "years (e.g. 10+ years, senior director, VP) are NOT a seniority fit — "
        "score 4 or below unless something else is genuinely exceptional."
    )
    return (
        f"Track: {track_name}\n{hint_block}"
        f"CANDIDATE RESUME SUMMARY:\n{resume_summary}\n\n"
        f"JOB POST (from LinkedIn):\n{jd}\n\n"
        f"Posted by: {author_title}\n\n"
        f"{yoe_clause}\n\n"
        "Score the fit 0-10 and give a one-sentence reason (<25 words).\n"
        "Consider: role title match, must-have skills, seniority, domain, "
        "archetype fit per the TRACK GUARDRAIL (if any).\n"
        "Penalize: hard location mismatches, seniority mismatches, unrelated "
        "domains, archetype mismatches."
    )


def score_one(
    track_name: str,
    resume_summary: str,
    post: dict[str, Any],
    client: Anthropic | None = None,
) -> dict[str, Any]:
    client = client or Anthropic(api_key=config.ANTHROPIC_API_KEY)
    jd = post.get("jd_text") or post.get("text") or ""
    author_title = post.get("author_title") or ""

    prompt = _rerank_prompt(track_name, resume_summary, jd, author_title)

    response = client.messages.create(
        model=config.CLAUDE_MODEL_LIGHT,
        max_tokens=256,
        tools=[
            {
                "name": "save_score",
                "description": "Record the fit score and reason.",
                "input_schema": SCORE_SCHEMA,
            }
        ],
        tool_choice={"type": "tool", "name": "save_score"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)

    return {"score": 0.0, "reason": "scoring failed"}


def rerank(
    track_name: str,
    resume_summary: str,
    posts: list[dict[str, Any]],
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    thr = score_threshold if score_threshold is not None else config.LLM_SCORE_THRESHOLD
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    kept: list[dict[str, Any]] = []
    for p in posts:
        result = score_one(track_name, resume_summary, p, client=client)
        scored = {**p, "llm_score": result.get("score", 0.0), "llm_reason": result.get("reason", "")}
        if scored["llm_score"] >= thr:
            kept.append(scored)
    kept.sort(key=lambda p: p["llm_score"], reverse=True)
    return kept


def assign_best_track(
    post: dict[str, Any],
    tracks: list[dict[str, Any]],
    client: Anthropic | None = None,
) -> str:
    """When a post matched multiple tracks' similarity gate, pick the single best track.
    Uses the LLM-produced vote. Fallback: track with highest embedding similarity.
    """
    client = client or Anthropic(api_key=config.ANTHROPIC_API_KEY)
    jd = post.get("jd_text") or post.get("text") or ""
    track_names = [t["track"] for t in tracks]
    prompt = (
        "Given this LinkedIn hiring post, pick which single career track it fits best.\n\n"
        f"Options: {', '.join(track_names)}\n\n"
        f"POST:\n{jd[:3000]}\n\n"
        "Pick exactly one track name from the Options list. No other text."
    )
    response = client.messages.create(
        model=config.CLAUDE_MODEL_LIGHT,
        max_tokens=32,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    ).strip()
    return text if text in track_names else track_names[0]
