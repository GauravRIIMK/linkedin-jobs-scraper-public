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
