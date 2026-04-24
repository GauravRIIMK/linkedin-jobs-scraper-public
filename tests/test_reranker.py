"""Reranker: score_one structured output, threshold filtering, best-track assignment."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src import reranker


def _fake_message(tool_input: dict):
    msg = MagicMock()
    block = MagicMock()
    block.type = "tool_use"
    block.input = tool_input
    msg.content = [block]
    return msg


def _fake_text_message(text: str):
    msg = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = text
    msg.content = [block]
    return msg


def test_score_one_returns_structured_output():
    client = MagicMock()
    client.messages.create.return_value = _fake_message({"score": 8.5, "reason": "strong fit"})
    result = reranker.score_one(
        "growth",
        "resume summary",
        {"jd_text": "Growth role in Bangalore"},
        client=client,
    )
    assert result["score"] == 8.5
    assert "strong" in result["reason"]


def test_score_one_defaults_on_malformed_response():
    client = MagicMock()
    # Claude returned only text, no tool_use
    msg = MagicMock()
    block = MagicMock()
    block.type = "text"
    msg.content = [block]
    client.messages.create.return_value = msg
    result = reranker.score_one("t", "r", {"jd_text": "x"}, client=client)
    assert result["score"] == 0.0
    assert "failed" in result["reason"]


def test_rerank_filters_by_threshold():
    posts = [
        {"jd_text": "a"},
        {"jd_text": "b"},
        {"jd_text": "c"},
    ]
    scores = [{"score": 9.0, "reason": "great"}, {"score": 5.0, "reason": "meh"}, {"score": 7.5, "reason": "ok"}]

    def fake_score_one(track, summary, post, client=None):
        return scores.pop(0)

    with patch("src.reranker.score_one", side_effect=fake_score_one):
        kept = reranker.rerank("growth", "summary", posts, score_threshold=7.0)
    assert len(kept) == 2
    # Sorted by score desc
    assert kept[0]["llm_score"] == 9.0
    assert kept[1]["llm_score"] == 7.5


def test_rerank_empty_input():
    assert reranker.rerank("t", "r", []) == []


def test_assign_best_track_uses_llm_vote():
    client = MagicMock()
    client.messages.create.return_value = _fake_text_message("growth_marketing")
    best = reranker.assign_best_track(
        {"jd_text": "Growth Marketer needed"},
        [{"track": "growth_marketing"}, {"track": "product_ai"}],
        client=client,
    )
    assert best == "growth_marketing"


def test_assign_best_track_falls_back_when_llm_hallucinates():
    client = MagicMock()
    client.messages.create.return_value = _fake_text_message("MadeUpTrack")
    best = reranker.assign_best_track(
        {"jd_text": "?"},
        [{"track": "growth_marketing"}, {"track": "product_ai"}],
        client=client,
    )
    # Should fall back to first candidate
    assert best == "growth_marketing"


def test_assign_best_track_strips_whitespace():
    client = MagicMock()
    client.messages.create.return_value = _fake_text_message("  product_ai  \n")
    best = reranker.assign_best_track(
        {"jd_text": "?"},
        [{"track": "growth_marketing"}, {"track": "product_ai"}],
        client=client,
    )
    assert best == "product_ai"


# ---------- Track-specific Haiku guardrails ----------
#
# Without a track hint Haiku scores posts in a generic "does JD overlap
# resume?" mode, which systematically rewards lexical overlap over
# archetype fit. For growth_marketing that means Growth Engineers / GTM
# Engineers with heavy "growth"/"revenue" prose score well alongside
# genuine growth marketers — the 2026-04-24 zero-match bug.
#
# These tests lock the hint in place so a refactor can't silently delete
# it and reintroduce the pathology.


def test_rerank_prompt_includes_growth_marketing_hint():
    """growth_marketing must have a track hint so Haiku knows the archetype."""
    prompt = reranker._rerank_prompt(
        "growth_marketing",
        "resume",
        "a growth marketing JD",
        "HR at Acme",
    )
    # The formatted block — 'TRACK GUARDRAIL:' with the colon — only
    # appears when a hint exists for the track.
    assert "TRACK GUARDRAIL:" in prompt
    # Positive archetype tokens — we want Haiku to score HIGH on these
    assert "Brand" in prompt or "brand" in prompt.lower()
    assert "Lifecycle" in prompt or "lifecycle" in prompt.lower()
    assert "Product Marketing" in prompt or "PMM" in prompt
    # Negative archetype tokens — we want Haiku to score LOW on these.
    # These are the exact roles that contaminated the top-N on 2026-04-24.
    assert "Growth Engineer" in prompt
    assert "GTM Engineer" in prompt


def test_rerank_prompt_has_hint_for_each_archetype_track():
    """Every track with an archetype mismatch pattern should have a hint.
    Currently: ai_ops_automation + growth_marketing. Add more as the
    pattern emerges for other tracks (e.g. strategy_consulting → 'consultant
    != consulting manager')."""
    assert "ai_ops_automation" in reranker._TRACK_HINTS
    assert "growth_marketing" in reranker._TRACK_HINTS


def test_rerank_prompt_omits_guardrail_block_for_tracks_without_hint():
    """Tracks without a hint must not have a formatted 'TRACK GUARDRAIL:'
    block with empty content dangling in the prompt — that would confuse
    Haiku and waste tokens. The instruction text references 'TRACK
    GUARDRAIL' unconditionally ('archetype fit per the TRACK GUARDRAIL
    (if any)'), but the block with the actual guardrail content only
    appears for tracks that have a hint."""
    prompt = reranker._rerank_prompt(
        "strategy_consulting",  # no hint defined
        "resume",
        "a JD",
        "HR at Acme",
    )
    # The formatted block header — with the colon — must NOT appear.
    assert "TRACK GUARDRAIL:" not in prompt
