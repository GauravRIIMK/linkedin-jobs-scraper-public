"""Human-in-the-loop role selection CLI. Runs interactively via `python -m src.setup`."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import config

console = Console()


def confirm_roles_for_track(track: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Show the user the parsed profile and let them pick which roles to hunt for."""
    console.print()
    console.print(
        Panel(
            f"[bold cyan]Track:[/] {track}\n"
            f"[bold]Name:[/] {profile.get('name', '?')}\n"
            f"[bold]Current title:[/] {profile.get('current_title', '?')}\n"
            f"[bold]Seniority:[/] {profile.get('seniority_level', '?')}  "
            f"[bold]Years:[/] {profile.get('years_experience', '?')}\n"
            f"[bold]Domains:[/] {', '.join(profile.get('primary_domains', []))}\n",
            title="Resume parsed",
            border_style="cyan",
        )
    )

    top_skills = profile.get("top_skills", [])
    if top_skills:
        tbl = Table(title="Top skills", show_header=False, border_style="dim")
        tbl.add_row(", ".join(top_skills[:20]))
        console.print(tbl)

    candidate_roles: list[str] = profile.get("candidate_role_titles", [])
    if not candidate_roles:
        console.print("[yellow]Warning: Claude returned no candidate role titles.[/]")
        candidate_roles = [profile.get("current_title", "")]

    selected: list[str] = questionary.checkbox(
        f"Which roles should this track hunt for? [{track}]",
        choices=[questionary.Choice(r, checked=True) for r in candidate_roles],
        instruction="(space to toggle, enter to confirm)",
    ).ask()

    if selected is None:
        raise SystemExit("Aborted.")

    extra = questionary.text(
        "Add any extra role titles (comma-separated), or leave blank:"
    ).ask()
    if extra and extra.strip():
        selected.extend([r.strip() for r in extra.split(",") if r.strip()])

    # City override per track (optional)
    default_cities = ", ".join(config.CITIES)
    city_answer = questionary.text(
        f"Cities for this track (comma-sep) [{default_cities}]:"
    ).ask()
    cities = (
        [c.strip() for c in city_answer.split(",") if c.strip()]
        if city_answer and city_answer.strip()
        else list(config.CITIES)
    )

    include_remote = questionary.confirm(
        "Include remote-only postings?", default=True
    ).ask()

    active = questionary.confirm(
        "Activate this track in weekly runs?", default=True
    ).ask()

    return {
        "track": track,
        "active": bool(active),
        "roles": selected,
        "cities": cities,
        "include_remote": bool(include_remote),
        "resume_summary": profile.get("resume_summary", ""),
        "confirmed_at": datetime.now(timezone.utc).isoformat(),
    }


def save_preferences(tracks: list[dict[str, Any]]) -> Path:
    config.PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tracks": tracks, "updated_at": datetime.now(timezone.utc).isoformat()}
    config.PREFS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return config.PREFS_PATH


def load_preferences() -> dict[str, Any]:
    if not config.PREFS_PATH.exists():
        raise FileNotFoundError(
            f"{config.PREFS_PATH} not found. Run `python -m src.setup` first."
        )
    # utf-8-sig: tolerates an optional leading BOM (PowerShell `Get-Content -Raw` adds one)
    return json.loads(config.PREFS_PATH.read_text(encoding="utf-8-sig"))
