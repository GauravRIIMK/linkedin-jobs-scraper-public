"""Command-line interface: setup | run | dry-run | status | apply | reject | tracks."""
from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console

from . import config, diagnose, pipeline, role_selector, tracking

console = Console()


def _cmd_setup(_: argparse.Namespace) -> int:
    pipeline.run_setup()
    return 0


def _cmd_run(_: argparse.Namespace) -> int:
    config.assert_secrets("APIFY_TOKEN", "ANTHROPIC_API_KEY", "VOYAGE_API_KEY")
    pipeline.run_pipeline(dry_run=False)
    return 0


def _cmd_dry_run(_: argparse.Namespace) -> int:
    config.assert_secrets("APIFY_TOKEN", "ANTHROPIC_API_KEY", "VOYAGE_API_KEY")
    pipeline.run_pipeline(dry_run=True)
    return 0


def _cmd_status(_: argparse.Namespace) -> int:
    pipeline.run_status()
    return 0


def _cmd_diagnose(_: argparse.Namespace) -> int:
    config.assert_secrets("APIFY_TOKEN", "ANTHROPIC_API_KEY", "VOYAGE_API_KEY")
    diagnose.main()
    return 0


def _cmd_apply(args: argparse.Namespace) -> int:
    ok = tracking.mark_applied(args.url)
    console.print("[green]marked applied[/]" if ok else "[red]failed[/]")
    return 0 if ok else 1


def _cmd_reject(args: argparse.Namespace) -> int:
    ok = tracking.mark_rejected(args.url)
    console.print("[green]marked rejected — will be skipped next week[/]" if ok else "[red]failed[/]")
    return 0 if ok else 1


def _cmd_tracks(args: argparse.Namespace) -> int:
    try:
        prefs = role_selector.load_preferences()
    except FileNotFoundError:
        console.print("[yellow]No preferences. Run `python -m src setup`.[/]")
        return 1

    if args.toggle:
        for t in prefs["tracks"]:
            if t["track"] == args.toggle:
                t["active"] = not t.get("active", True)
                role_selector.save_preferences(prefs["tracks"])
                console.print(f"[green]{args.toggle}: active = {t['active']}[/]")
                return 0
        console.print(f"[red]Track {args.toggle!r} not found.[/]")
        return 1

    console.print(json.dumps(prefs, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="LinkedIn Jobs Scraper — find #hiring posts with Google Forms, matched to your resume.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="Parse resume PDFs and confirm roles per track (interactive).").set_defaults(func=_cmd_setup)
    sub.add_parser("run", help="Full weekly scan: scrape + match + email digest.").set_defaults(func=_cmd_run)
    sub.add_parser("dry-run", help="Full scan without email / without marking posts as seen.").set_defaults(func=_cmd_dry_run)
    sub.add_parser("status", help="Show tracking stats, track config, recent run logs.").set_defaults(func=_cmd_status)
    sub.add_parser("diagnose", help="API health checks + verbose pipeline per track (shows every Haiku score + reason).").set_defaults(func=_cmd_diagnose)

    p_apply = sub.add_parser("apply", help="Mark a post URL as applied (suppresses it next week).")
    p_apply.add_argument("url")
    p_apply.set_defaults(func=_cmd_apply)

    p_reject = sub.add_parser("reject", help="Mark a post URL as rejected (suppresses it next week).")
    p_reject.add_argument("url")
    p_reject.set_defaults(func=_cmd_reject)

    p_tracks = sub.add_parser("tracks", help="List tracks, or --toggle <track> to activate/deactivate.")
    p_tracks.add_argument("--toggle", metavar="TRACK")
    p_tracks.set_defaults(func=_cmd_tracks)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
