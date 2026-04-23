"""Matcher: cosine, ordering, threshold filtering, empty inputs."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import matcher


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(matcher.cosine(a, b)) < 1e-6


def test_cosine_identical():
    v = np.array([3.0, 4.0, 5.0])
    assert matcher.cosine(v, v) == pytest_approx(1.0)


def test_cosine_zero_vector_no_crash():
    a = np.zeros(5)
    b = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    # Should not divide-by-zero
    result = matcher.cosine(a, b)
    assert result == 0.0


def test_score_posts_for_track_empty_returns_empty():
    assert matcher.score_posts_for_track("resume", []) == []


def test_score_posts_attaches_similarity_and_sorts():
    posts = [
        {"jd_text": "Growth marketer B"},
        {"jd_text": "Growth marketer A"},
    ]

    def fake_embed(texts, input_type):
        # Post 0 will be similar to resume; post 1 dissimilar
        if input_type == "query":
            return np.array([[1.0, 0.0]], dtype=np.float32)
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with patch("src.matcher.embed_texts", side_effect=fake_embed):
        out = matcher.score_posts_for_track("resume", posts)
    assert len(out) == 2
    assert out[0]["similarity"] > out[1]["similarity"]


def test_filter_and_rerank_target_respects_threshold():
    scored = [
        {"jd_text": "x", "similarity": 0.9},
        {"jd_text": "y", "similarity": 0.5},
        {"jd_text": "z", "similarity": 0.7},
    ]
    out = matcher.filter_and_rerank_target(scored, similarity_threshold=0.6, top_n=10)
    assert len(out) == 2
    assert all(p["similarity"] >= 0.6 for p in out)


def test_filter_and_rerank_top_n_cap():
    scored = [{"similarity": 0.9, "jd_text": f"p{i}"} for i in range(50)]
    out = matcher.filter_and_rerank_target(scored, similarity_threshold=0.1, top_n=5)
    assert len(out) == 5


# ---------- source-aware quota ----------
#
# LinkedIn posts (free-form recruiter chatter) systematically lose the raw
# cosine-similarity race to dense machine-generated JDs from Naukri /
# LinkedIn Easy-Apply listings — even when the LinkedIn post carries far
# stronger recruiter intent. The quota reserves N slots in the top-K so
# a track dominated by job-board listings can't starve out LinkedIn posts.


def test_filter_and_rerank_reserves_quota_for_linkedin_posts():
    """Even when job-board listings dominate on similarity, the LinkedIn-post
    quota must be honored: top slots are reserved regardless of rank."""
    scored = [
        # 20 high-similarity Naukri listings — would otherwise saturate the top-N
        *[
            {"similarity": 0.95 - i * 0.001, "source": "naukri_job", "jd_text": f"n{i}"}
            for i in range(20)
        ],
        # 5 lower-similarity LinkedIn posts
        *[
            {"similarity": 0.70 - i * 0.01, "source": "linkedin_post", "jd_text": f"l{i}"}
            for i in range(5)
        ],
    ]
    out = matcher.filter_and_rerank_target(
        scored, similarity_threshold=0.6, top_n=10, min_linkedin_posts=3
    )
    # 3 quota slots + 7 remaining = 10 total
    assert len(out) == 10
    ln_count = sum(1 for p in out if p.get("source") == "linkedin_post")
    assert ln_count == 3
    # The 3 reserved LinkedIn posts should be the top-3 by similarity within their source
    reserved_ln = [p for p in out if p.get("source") == "linkedin_post"]
    reserved_sims = [p["similarity"] for p in reserved_ln]
    assert reserved_sims == sorted(reserved_sims, reverse=True)


def test_filter_and_rerank_quota_degrades_gracefully_when_no_linkedin_posts():
    """No LinkedIn posts above threshold → quota is unfilled, other sources
    absorb the full top_n. Zero empty slots wasted."""
    scored = [
        {"similarity": 0.9, "source": "naukri_job", "jd_text": f"n{i}"}
        for i in range(20)
    ]
    out = matcher.filter_and_rerank_target(
        scored, similarity_threshold=0.5, top_n=10, min_linkedin_posts=5
    )
    assert len(out) == 10  # Full slate, no wasted slots
    assert all(p["source"] == "naukri_job" for p in out)


def test_filter_and_rerank_quota_caps_at_eligible_linkedin_count():
    """Only 2 LinkedIn posts clear the threshold but quota=5 → we don't fabricate;
    fill the remaining 8 slots from other sources."""
    scored = [
        {"similarity": 0.95, "source": "linkedin_post", "jd_text": "l0"},
        {"similarity": 0.90, "source": "linkedin_post", "jd_text": "l1"},
        {"similarity": 0.50, "source": "linkedin_post", "jd_text": "l2-below-thr"},
        *[
            {"similarity": 0.80 - i * 0.001, "source": "naukri_job", "jd_text": f"n{i}"}
            for i in range(15)
        ],
    ]
    out = matcher.filter_and_rerank_target(
        scored, similarity_threshold=0.6, top_n=10, min_linkedin_posts=5
    )
    assert len(out) == 10
    ln_count = sum(1 for p in out if p.get("source") == "linkedin_post")
    assert ln_count == 2  # Only 2 linkedin_posts above threshold — no phantom fill
    nk_count = sum(1 for p in out if p.get("source") == "naukri_job")
    assert nk_count == 8


def test_filter_and_rerank_quota_zero_disables_reservation():
    """min_linkedin_posts=0 → pure top-N on similarity, no quota machinery."""
    scored = [
        {"similarity": 0.95, "source": "naukri_job", "jd_text": "n0"},
        {"similarity": 0.90, "source": "linkedin_post", "jd_text": "l0"},
        {"similarity": 0.85, "source": "naukri_job", "jd_text": "n1"},
    ]
    out = matcher.filter_and_rerank_target(
        scored, similarity_threshold=0.5, top_n=2, min_linkedin_posts=0
    )
    assert len(out) == 2
    # Strict top-2 by similarity: the 0.95 naukri + the 0.90 linkedin
    assert out[0]["jd_text"] == "n0"
    assert out[1]["jd_text"] == "l0"


def test_filter_and_rerank_quota_larger_than_top_n_is_clamped():
    """Config footgun: quota=50 but top_n=10 — the quota shouldn't overflow
    the output. Clamp to top_n so a bad config doesn't amplify the bug."""
    scored = [
        {"similarity": 0.9 - i * 0.001, "source": "linkedin_post", "jd_text": f"l{i}"}
        for i in range(30)
    ]
    out = matcher.filter_and_rerank_target(
        scored, similarity_threshold=0.5, top_n=10, min_linkedin_posts=50
    )
    assert len(out) == 10  # Not 50


def test_filter_and_rerank_quota_preserves_similarity_order_within_reserve():
    """Reserved LinkedIn posts must still be ordered highest-similarity first —
    the rerank pool's position shouldn't be randomized by the quota."""
    scored = [
        {"similarity": 0.65, "source": "linkedin_post", "jd_text": "low-ln"},
        {"similarity": 0.85, "source": "linkedin_post", "jd_text": "high-ln"},
        {"similarity": 0.75, "source": "linkedin_post", "jd_text": "mid-ln"},
        {"similarity": 0.90, "source": "naukri_job", "jd_text": "top-nk"},
    ]
    out = matcher.filter_and_rerank_target(
        scored, similarity_threshold=0.6, top_n=4, min_linkedin_posts=3
    )
    # All 4 present, LinkedIn posts appear first (reserved slots), high→low within.
    ln_slice = [p for p in out if p["source"] == "linkedin_post"]
    assert [p["jd_text"] for p in ln_slice] == ["high-ln", "mid-ln", "low-ln"]


# pytest.approx helper
def pytest_approx(v):
    import pytest
    return pytest.approx(v, abs=1e-6)


# ---------- rate-limit retry behavior ----------


def _fake_voyage_client(embed_side_effect):
    """Factory for a fake voyageai.Client whose .embed() runs the given side_effect."""
    client = MagicMock()
    client.embed = MagicMock(side_effect=embed_side_effect)
    return client


def test_embed_texts_retries_on_rate_limit_then_succeeds():
    """A transient 'rate limit' error should trigger a sleep + retry."""
    fake_result = MagicMock(embeddings=[[0.1, 0.2]])
    calls = {"n": 0}

    def embed(texts, model, input_type):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("rate limit: reduced rate limits of 3 RPM")
        return fake_result

    with patch("src.matcher._client", return_value=_fake_voyage_client(embed)), \
         patch("src.matcher.time.sleep") as mock_sleep:
        out = matcher.embed_texts(["hello"], "document")

    assert out.shape == (1, 2)
    assert calls["n"] == 3
    # Two retries → two sleeps
    assert mock_sleep.call_count == 2
    # Each delay is >= 25s (per-minute window)
    for call in mock_sleep.call_args_list:
        assert call.args[0] >= 25


def test_embed_texts_does_not_retry_on_non_rate_limit_error():
    """Auth / network errors should raise immediately, not mask real bugs."""
    def embed(texts, model, input_type):
        raise RuntimeError("invalid api key")

    with patch("src.matcher._client", return_value=_fake_voyage_client(embed)), \
         patch("src.matcher.time.sleep") as mock_sleep:
        with pytest.raises(RuntimeError, match="invalid api key"):
            matcher.embed_texts(["hello"], "document")

    mock_sleep.assert_not_called()


def test_embed_texts_gives_up_after_max_attempts():
    """Persistent rate-limit should eventually bubble up, not spin forever."""
    def embed(texts, model, input_type):
        raise RuntimeError("429 Too Many Requests")

    with patch("src.matcher._client", return_value=_fake_voyage_client(embed)), \
         patch("src.matcher.time.sleep"):
        with pytest.raises(RuntimeError, match="429"):
            matcher.embed_texts(["hello"], "document", max_attempts=3)


def test_is_rate_limit_error_matches_common_phrasings():
    """The sniffer must catch the shapes the Voyage SDK emits in the wild."""
    assert matcher._is_rate_limit_error(RuntimeError("Rate limit exceeded"))
    assert matcher._is_rate_limit_error(RuntimeError("429 Too Many Requests"))
    assert matcher._is_rate_limit_error(
        RuntimeError("You have not yet added your payment method ... reduced rate limits")
    )
    assert matcher._is_rate_limit_error(RuntimeError("Quota exceeded"))
    assert not matcher._is_rate_limit_error(RuntimeError("Invalid API key"))
    assert not matcher._is_rate_limit_error(ValueError("bad input"))


# ---------- resume embedding cache ----------


def test_embed_resume_cached_hits_voyage_on_first_call_only(tmp_path, monkeypatch):
    """Second call with the same summary must reuse the cached vector."""
    monkeypatch.setattr(matcher.config, "RESUME_EMBEDDING_CACHE", tmp_path / "cache.json")

    calls = {"n": 0}

    def fake_embed(texts, input_type):
        calls["n"] += 1
        return np.array([[0.5, 0.5]], dtype=np.float32)

    with patch("src.matcher.embed_texts", side_effect=fake_embed):
        v1 = matcher.embed_resume_cached("my summary")
        v2 = matcher.embed_resume_cached("my summary")

    assert calls["n"] == 1  # second call served from cache
    assert np.allclose(v1, v2)


def test_embed_resume_cached_invalidates_when_summary_changes(tmp_path, monkeypatch):
    """Editing the resume → new hash → forces a fresh embed."""
    monkeypatch.setattr(matcher.config, "RESUME_EMBEDDING_CACHE", tmp_path / "cache.json")

    calls = {"n": 0}

    def fake_embed(texts, input_type):
        calls["n"] += 1
        return np.array([[float(calls["n"]), 0.0]], dtype=np.float32)

    with patch("src.matcher.embed_texts", side_effect=fake_embed):
        v1 = matcher.embed_resume_cached("version 1")
        v2 = matcher.embed_resume_cached("version 2")

    assert calls["n"] == 2
    assert not np.allclose(v1, v2)


def test_embed_resume_cached_survives_corrupt_cache_file(tmp_path, monkeypatch):
    """A garbage cache file shouldn't crash the pipeline — just rebuild it."""
    cache_path = tmp_path / "cache.json"
    cache_path.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(matcher.config, "RESUME_EMBEDDING_CACHE", cache_path)

    def fake_embed(texts, input_type):
        return np.array([[1.0, 0.0]], dtype=np.float32)

    with patch("src.matcher.embed_texts", side_effect=fake_embed):
        v = matcher.embed_resume_cached("summary")

    assert v.shape == (2,)
    # Cache should now be valid JSON with one entry
    import json
    reloaded = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(reloaded) == 1


def test_score_posts_for_track_uses_resume_cache(tmp_path, monkeypatch):
    """Integration: score_posts_for_track must route the resume through the
    cache so repeated runs of the same track don't burn Voyage calls."""
    monkeypatch.setattr(matcher.config, "RESUME_EMBEDDING_CACHE", tmp_path / "cache.json")

    def fake_embed(texts, input_type):
        if input_type == "query":
            return np.array([[1.0, 0.0]], dtype=np.float32)
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    with patch("src.matcher.embed_texts", side_effect=fake_embed) as m:
        matcher.score_posts_for_track("resume", [{"jd_text": "x"}])
        first_call_count = m.call_count
        matcher.score_posts_for_track("resume", [{"jd_text": "y"}])

    # First call: 1 query embed + 1 document embed = 2.
    # Second call: only the document embed — resume is cached → 3 total.
    assert first_call_count == 2
    assert m.call_count == 3


# ---------- batch chunking (Voyage 1000-doc / 320K-token caps) ----------


def test_estimate_tokens_clamps_at_per_doc_cap():
    """One pathological doc must not blow the whole batch budget —
    Voyage truncates to 32K server-side, so we cap our estimate there too."""
    huge = "x" * 10_000_000  # naive est ≈ 2.5M tokens
    assert matcher._estimate_tokens(huge) == matcher._VOYAGE_MAX_TOKENS_PER_DOC


def test_estimate_tokens_never_zero():
    """Empty/tiny strings must not produce 0 — batch packing adds to a
    running sum and needs monotonic progress."""
    assert matcher._estimate_tokens("") >= 1
    assert matcher._estimate_tokens("x") >= 1


def test_pack_batches_single_batch_under_caps():
    """Small input that comfortably fits should be a single batch."""
    texts = ["short doc"] * 10
    batches = matcher._pack_batches(texts)
    assert batches == [(0, 10)]


def test_pack_batches_splits_at_doc_cap():
    """300 short docs → 3 batches at the 128-doc boundary (128 / 128 / 44)."""
    texts = ["x"] * 300
    batches = matcher._pack_batches(texts)
    sizes = [e - s for s, e in batches]
    assert sum(sizes) == 300  # no docs lost
    assert max(sizes) <= matcher._VOYAGE_MAX_DOCS_PER_BATCH
    # Consecutive slices — no gaps, no overlaps
    for (s1, e1), (s2, e2) in zip(batches, batches[1:]):
        assert e1 == s2


def test_pack_batches_splits_at_token_cap():
    """Chatty docs can trip the token cap before the doc cap does."""
    # ~32K chars → ~8K tokens per doc. 40 such docs → ~320K tokens > 280K cap.
    fat = "x" * 32_000
    texts = [fat] * 40
    batches = matcher._pack_batches(texts)
    assert len(batches) >= 2
    # Each batch stays under the token cap
    for s, e in batches:
        batch_tokens = sum(matcher._estimate_tokens(t) for t in texts[s:e])
        assert batch_tokens <= matcher._VOYAGE_MAX_TOKENS_PER_BATCH


def test_pack_batches_single_oversized_doc_still_produces_one_batch():
    """A doc that alone exceeds the batch cap must still be embeddable —
    Voyage truncates at 32K tokens server-side; we don't bail out."""
    monster = "x" * 50_000_000
    batches = matcher._pack_batches([monster])
    assert batches == [(0, 1)]


def test_pack_batches_empty_input():
    """Edge: empty list should not crash. embed_texts short-circuits before
    calling this, but defending the helper keeps it reusable."""
    assert matcher._pack_batches([]) == [(0, 0)]


def test_embed_texts_multi_batch_preserves_order_and_concatenates():
    """When chunking kicks in, the stitched output row order must match
    input order — otherwise post↔embedding alignment breaks downstream."""
    texts = [f"doc-{i}" for i in range(250)]  # forces 2 batches at 128 cap

    def embed(batch_texts, model, input_type):
        # Encode each doc's index into its vector so order is verifiable.
        idxs = [int(t.split("-")[1]) for t in batch_texts]
        return MagicMock(embeddings=[[float(i), 0.0] for i in idxs])

    with patch("src.matcher._client", return_value=_fake_voyage_client(embed)):
        out = matcher.embed_texts(texts, "document")

    assert out.shape == (250, 2)
    # Row i must encode doc index i — proves concatenation preserved order.
    for i in range(250):
        assert out[i][0] == float(i)


def test_embed_texts_single_batch_skips_concatenation():
    """Fast path: when everything fits in one call, don't waste a concat."""
    fake_result = MagicMock(embeddings=[[0.1, 0.2]] * 5)
    call_count = {"n": 0}

    def embed(texts, model, input_type):
        call_count["n"] += 1
        return fake_result

    with patch("src.matcher._client", return_value=_fake_voyage_client(embed)):
        out = matcher.embed_texts(["hi"] * 5, "document")

    assert out.shape == (5, 2)
    assert call_count["n"] == 1  # single Voyage call


def test_embed_texts_empty_returns_zero_shape():
    """Caller passing [] is almost certainly a bug — an empty array raises
    loudly on any downstream matrix op instead of silently failing."""
    out = matcher.embed_texts([], "document")
    assert out.shape == (0, 0)


def test_embed_texts_chunks_real_world_track_size():
    """Regression: a track the size of ai_ops_automation (1206 docs, chatty
    JDs) used to crash with 'batch size 1091 > 1000'. Verify we chunk it."""
    # 1200 docs, each ~8K chars → ~2K tokens. Total ~2.4M tokens → many batches.
    texts = ["x" * 8_000 for _ in range(1200)]
    call_batches: list[int] = []

    def embed(batch_texts, model, input_type):
        call_batches.append(len(batch_texts))
        return MagicMock(embeddings=[[0.0, 0.0]] * len(batch_texts))

    with patch("src.matcher._client", return_value=_fake_voyage_client(embed)):
        out = matcher.embed_texts(texts, "document")

    assert out.shape == (1200, 2)
    assert len(call_batches) >= 2
    # Every call stays under Voyage's hard caps.
    for n in call_batches:
        assert n <= matcher._VOYAGE_MAX_DOCS_PER_BATCH
    assert sum(call_batches) == 1200  # nothing dropped
