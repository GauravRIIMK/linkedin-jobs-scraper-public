"""Semantic similarity: Voyage-3 embeddings + cosine similarity."""
from __future__ import annotations

import hashlib
import json
import random
import time
from typing import Any

import numpy as np
import voyageai

from . import config


def _client() -> voyageai.Client:
    return voyageai.Client(api_key=config.VOYAGE_API_KEY)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Detect rate-limit errors across voyageai SDK versions.

    The SDK raises different exception types across releases; we sniff the
    error message instead of pinning a specific class. Also catches the
    free-tier "reduced rate limits of 3 RPM" message Voyage sends when no
    payment method is on file.
    """
    msg = f"{type(exc).__name__}: {exc}".lower()
    return (
        "rate limit" in msg
        or "ratelimit" in msg
        or "429" in msg
        or "too many requests" in msg
        or "reduced rate limits" in msg
        or "quota" in msg
    )


# ---------------------------------------------------------------------------
# Voyage batch-size caps
#
# Voyage enforces BOTH limits per embed call and rejects the whole batch if
# either is exceeded:
#   - 1000 documents
#   - 320_000 tokens (post per-doc truncation at 32K tokens each)
#
# At multi-source scale we now routinely produce 1000-2000 filtered posts
# per track — a single naive embed call fails on doc count first, or on
# token count second for chatty job descriptions. We chunk with safety
# margins and re-assemble in input order below.
# ---------------------------------------------------------------------------

_VOYAGE_MAX_DOCS_PER_BATCH = 128
_VOYAGE_MAX_TOKENS_PER_BATCH = 280_000  # 12% under the 320K ceiling
# Voyage truncates each doc to 32K tokens server-side; we cap our estimate
# at the same value so one pathologically long doc can't blow the budget.
_VOYAGE_MAX_TOKENS_PER_DOC = 32_000


def _estimate_tokens(text: str) -> int:
    """Cheap ~4-chars-per-token estimate, clamped to Voyage's per-doc cap.

    Good enough for batch packing — we already leave 12% headroom below
    the hard limit. If a future model changes the tokenization ratio this
    may need retuning.
    """
    est = max(len(text) // 4, 1)
    return min(est, _VOYAGE_MAX_TOKENS_PER_DOC)


def _pack_batches(texts: list[str]) -> list[tuple[int, int]]:
    """Greedy-pack texts into (start, end) slice tuples under Voyage's caps.

    Always produces at least one batch even for a single oversized text
    (Voyage's server-side truncation handles the per-doc part).
    """
    batches: list[tuple[int, int]] = []
    start = 0
    tokens = 0
    for i, t in enumerate(texts):
        tk = _estimate_tokens(t)
        # Starting a new batch: current batch is full AND we've got at least
        # one text in it (the `i > start` guard prevents an infinite-zero-size
        # batch if the first text on its own already exceeds a cap).
        if i > start and (
            (i - start) >= _VOYAGE_MAX_DOCS_PER_BATCH
            or (tokens + tk) > _VOYAGE_MAX_TOKENS_PER_BATCH
        ):
            batches.append((start, i))
            start = i
            tokens = 0
        tokens += tk
    batches.append((start, len(texts)))
    return batches


def _embed_one_batch(
    texts: list[str],
    input_type: str,
    max_attempts: int,
) -> np.ndarray:
    """Call Voyage once with the given batch, retrying on rate-limit errors.

    Free tier is 3 RPM / 10K TPM: in a multi-track run the 4th or 5th call
    can land in an already-full per-minute window. Wait ~25-70s (windows
    are per-minute so shorter backoff is wasteful) and retry. Non-rate-
    limit errors re-raise immediately — we don't want to paper over
    auth/network bugs.
    """
    client = _client()
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            result = client.embed(texts, model=config.VOYAGE_MODEL, input_type=input_type)
            return np.array(result.embeddings, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 — SDK raises varied types
            last_exc = exc
            if not _is_rate_limit_error(exc) or attempt == max_attempts - 1:
                raise
            delay = 25 + attempt * 15 + random.uniform(0, 5)
            time.sleep(delay)
    raise RuntimeError("embed_texts: exhausted retries") from last_exc


def embed_texts(
    texts: list[str],
    input_type: str,
    max_attempts: int = 5,
) -> np.ndarray:
    """Embed a list of texts, chunking to stay under Voyage's per-call caps.

    input_type is 'document' or 'query' per Voyage conventions.

    For inputs small enough to fit in one call (the common case for a
    single query), we skip the concatenation path entirely.
    """
    if not texts:
        # No sensible dim to return without asking Voyage; callers that
        # pass [] are almost certainly calling us in error. An empty array
        # is safest — it raises clearly on any downstream matrix op.
        return np.zeros((0, 0), dtype=np.float32)

    batches = _pack_batches(texts)
    if len(batches) == 1:
        return _embed_one_batch(texts, input_type, max_attempts)

    pieces: list[np.ndarray] = []
    for start, end in batches:
        pieces.append(_embed_one_batch(texts[start:end], input_type, max_attempts))
    return np.concatenate(pieces, axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / max(np.linalg.norm(a), 1e-10)
    b = b / max(np.linalg.norm(b), 1e-10)
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Resume embedding cache
#
# Per-track resume summaries change only when the user re-runs `setup` after
# editing the PDF. Voyage charges (and rate-limits) every embed call, so
# re-embedding the same summary N runs in a row is pure waste. We cache by
# SHA-256 of the summary text: edit the resume → new hash → auto-invalidates.
#
# File format (JSON):
#     { "<sha256>": [float, float, ...], ... }
# Vectors are stored as plain lists so the cache is human-inspectable and
# survives Python/NumPy version churn.
# ---------------------------------------------------------------------------

def _cache_key(summary: str) -> str:
    return hashlib.sha256(summary.encode("utf-8")).hexdigest()


def _load_resume_cache() -> dict[str, list[float]]:
    path = config.RESUME_EMBEDDING_CACHE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Corrupt cache shouldn't crash the run — just rebuild it.
        return {}


def _save_resume_cache(cache: dict[str, list[float]]) -> None:
    path = config.RESUME_EMBEDDING_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache), encoding="utf-8")


def embed_resume_cached(summary: str) -> np.ndarray:
    """Return the resume-summary embedding, hitting Voyage only on cache miss."""
    key = _cache_key(summary)
    cache = _load_resume_cache()
    if key in cache:
        return np.array(cache[key], dtype=np.float32)

    emb = embed_texts([summary], input_type="query")[0]
    cache[key] = emb.tolist()
    _save_resume_cache(cache)
    return emb


def score_posts_for_track(
    resume_summary: str,
    posts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach a similarity score to each post; return sorted desc by score."""
    if not posts:
        return []

    jd_texts = [p.get("jd_text") or p.get("text") or "" for p in posts]
    # Resume embedding is cached across runs (key = SHA256 of summary); JDs
    # are always fresh because the posts are always new.
    resume_emb = embed_resume_cached(resume_summary)
    jd_embs = embed_texts(jd_texts, input_type="document")

    scored: list[dict[str, Any]] = []
    for post, emb in zip(posts, jd_embs):
        s = cosine(resume_emb, emb)
        scored.append({**post, "similarity": s})
    scored.sort(key=lambda p: p["similarity"], reverse=True)
    return scored


def filter_and_rerank_target(
    scored_posts: list[dict[str, Any]],
    similarity_threshold: float | None = None,
    top_n: int | None = None,
    min_linkedin_posts: int | None = None,
) -> list[dict[str, Any]]:
    """Keep posts above the similarity threshold, cap to top N for LLM rerank.

    Source-aware quota: LinkedIn posts (source=='linkedin_post') get RESERVED
    slots in the top-N so they can't be crowded out by machine-generated
    job-board JDs that match the resume summary's JD-speak lexically but
    carry less recruiter intent.

    Why: LinkedIn hiring posts are chatty free-form text (hashtags, emojis,
    first-person voice); Naukri / LinkedIn-Jobs listings are dense, structured
    JDs. Pure cosine similarity systematically under-scores the former. With
    raw top-N, a track with hundreds of job-board listings can saturate all
    30 rerank slots with zero LinkedIn posts — exactly the outcome the user
    flagged when the final digest was 100% Naukri + Easy-Apply links.

    Algorithm:
      1. Keep all posts with similarity >= threshold (these are the
         "eligible" candidates).
      2. Reserve up to `min_linkedin_posts` slots for the highest-similarity
         linkedin_post entries (still sorted by similarity within the quota).
      3. Fill the remaining slots with the highest-similarity posts from any
         source that weren't already reserved.
      4. Total output: at most `top_n` posts, with at least `min(min_linkedin_posts, #eligible LinkedIn posts)` of them being LinkedIn posts.

    Graceful degradation: if no LinkedIn posts clear the similarity threshold,
    the quota is unfilled and the other slots absorb the full top_n — no
    empty slots wasted.
    """
    thr = similarity_threshold if similarity_threshold is not None else config.SIMILARITY_THRESHOLD
    top_n = top_n or config.LLM_RERANK_TOP_N
    if min_linkedin_posts is None:
        min_linkedin_posts = config.LLM_RERANK_MIN_LINKEDIN_POSTS

    # Sort defensively: production callers pass already-sorted output from
    # score_posts_for_track, but the quota's correctness (reserving the TOP
    # linkedin_posts, not arbitrary ones) depends on this order. Keeping the
    # sort here makes the function safe in isolation — cheap for realistic
    # input sizes (hundreds of posts per track).
    above = sorted(
        (p for p in scored_posts if p["similarity"] >= thr),
        key=lambda p: p["similarity"],
        reverse=True,
    )
    if not above:
        return []

    # Cap the reserve to top_n so a large quota doesn't overflow the output.
    quota = max(0, min(min_linkedin_posts, top_n))

    # `above` is now guaranteed desc-similarity; filter preserves that, so
    # first `quota` LinkedIn posts here are the highest-similarity ones.
    reserved = [p for p in above if p.get("source") == "linkedin_post"][:quota]
    reserved_ids = {id(p) for p in reserved}
    remaining = [p for p in above if id(p) not in reserved_ids]
    # `remaining` inherits desc-similarity order from `above`; no re-sort needed.

    slots_left = max(top_n - len(reserved), 0)
    return reserved + remaining[:slots_left]
