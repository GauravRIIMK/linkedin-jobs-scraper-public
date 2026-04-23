"""Parse a PDF resume via Claude native PDF input + structured JSON output."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from anthropic import Anthropic

from . import config

RESUME_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "current_title": {"type": "string"},
        "years_experience": {"type": "number"},
        "seniority_level": {
            "type": "string",
            "enum": ["entry", "mid", "senior", "lead", "principal", "executive"],
        },
        "primary_domains": {
            "type": "array",
            "items": {"type": "string"},
            "description": "e.g. SaaS, Fintech, E-commerce, EdTech, Healthcare",
        },
        "top_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "15-25 concrete skills (tools, frameworks, methodologies) ranked by prominence in the resume",
        },
        "candidate_role_titles": {
            "type": "array",
            "items": {"type": "string"},
            "description": "8-12 specific job titles that best fit this resume. Mix exact matches and adjacent roles. Examples: 'Growth Marketing Manager', 'Performance Marketing Lead', 'Retention Marketing Manager'",
        },
        "preferred_industries": {"type": "array", "items": {"type": "string"}},
        "resume_summary": {
            "type": "string",
            "description": "A dense 250-400 word summary covering role history, achievements, domains, tools. Optimized for semantic-search retrieval.",
        },
    },
    "required": [
        "name",
        "current_title",
        "candidate_role_titles",
        "top_skills",
        "resume_summary",
    ],
    "additionalProperties": False,
}


def analyze_resume(pdf_path: Path, client: Anthropic | None = None) -> dict[str, Any]:
    """Parse one PDF resume into a structured profile using Claude."""
    client = client or Anthropic(api_key=config.ANTHROPIC_API_KEY)
    pdf_bytes = pdf_path.read_bytes()
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("ascii")

    response = client.messages.create(
        model=config.CLAUDE_MODEL_HEAVY,
        max_tokens=4096,
        tools=[
            {
                "name": "save_resume_profile",
                "description": "Save the extracted resume profile. Be specific and comprehensive.",
                "input_schema": RESUME_SCHEMA,
            }
        ],
        tool_choice={"type": "tool", "name": "save_resume_profile"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract a structured profile from this resume. "
                            "For candidate_role_titles, list 8-12 specific job titles that match this resume, "
                            "including adjacent roles the person could realistically apply for. "
                            "Use Indian job-market conventions where relevant (e.g., 'Growth Manager', "
                            "'Business Development Lead', 'Product Marketing Manager')."
                        ),
                    },
                ],
            }
        ],
    )

    for block in response.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)

    raise RuntimeError("Claude did not return a structured resume profile.")


def track_name_from_filename(pdf_path: Path) -> str:
    """Infer track slug from filename. e.g. Resume_Growth_Marketing.pdf -> growth_marketing."""
    stem = pdf_path.stem
    parts = stem.split("_")
    # Drop 'Resume' + first name, keep remainder; drop trailing '2page'
    clean = [p for p in parts[2:] if p.lower() != "2page"]
    slug = "_".join(clean).lower() if clean else stem.lower()
    return slug or "default"


def prefer_longer_variant(pdf_paths: list[Path]) -> list[Path]:
    """If both 1-page and 2-page variants exist for a track, keep the 2-page (richer content)."""
    by_track: dict[str, Path] = {}
    for p in pdf_paths:
        track = track_name_from_filename(p)
        existing = by_track.get(track)
        if existing is None or p.stat().st_size > existing.stat().st_size:
            by_track[track] = p
    return list(by_track.values())


def save_profile(track: str, profile: dict[str, Any]) -> Path:
    path = Path(str(config.PROFILE_PATH_TMPL).format(track=track))
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_profile(track: str) -> dict[str, Any]:
    path = Path(str(config.PROFILE_PATH_TMPL).format(track=track))
    return json.loads(path.read_text(encoding="utf-8"))
