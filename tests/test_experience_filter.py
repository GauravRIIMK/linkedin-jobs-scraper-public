"""YoE extraction + filter behavior.

The existence bar for each pattern is: 'a real-world Indian tech JD the user
would encounter'. The corpus of failing examples from dry-run logs informs
these cases — e.g. 'Looking for 10+ years of experience' phrasing slipped
through previously because the old pipeline had no YoE gate at all.
"""
from __future__ import annotations

import pytest

from src import config, experience_filter


# --------------------- extract_min_yoe_required --------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("We're looking for 10+ years of experience.", 10),
        ("10+ years experience in digital marketing.", 10),
        ("Minimum 8 years of relevant experience required.", 8),
        ("At least 12 yrs of industry experience.", 12),
        ("Atleast 5 years in a senior role.", 5),
        ("5-10 years of experience in growth.", 5),
        ("5 to 10 years experience.", 5),
        ("5–10 years experience.", 5),      # en dash
        ("5—10 years experience.", 5),      # em dash
        ("Experience: 8+ years in AI.", 8),
        ("Exp: 6 years in digital.", 6),
        ("8+ yrs of hands-on ML work.", 8),
        # Multiple requirements: take the MAX of minimums.
        ("Need 2+ years Python AND 10+ years leadership experience.", 10),
        # Skill-only mentions shouldn't fool the gate — need a role-context word nearby.
        ("2+ years of Python.", None),  # no exp/experience/required keyword nearby
    ],
)
def test_extract_min_yoe_required(text, expected):
    assert experience_filter.extract_min_yoe_required(text) == (float(expected) if expected is not None else None)


def test_extract_returns_none_for_empty():
    assert experience_filter.extract_min_yoe_required("") is None
    assert experience_filter.extract_min_yoe_required("No year info here.") is None


def test_ignores_out_of_range_numbers():
    # "2024 years" → parsed as 2024 but clamped out; "150 years" → out; "30 years" → out.
    # These are almost always date/year references, not role requirements.
    assert experience_filter.extract_min_yoe_required("Since 2024 we have hired many.") is None
    assert experience_filter.extract_min_yoe_required("30+ years of industry experience in the firm.") is None


# ----------------------------- passes_yoe_filter ------------------------------

def _post(jd: str) -> dict:
    return {"jd_text": jd}


def test_passes_when_no_yoe_mentioned():
    assert experience_filter.passes_yoe_filter(_post("Hiring a growth marketer for Bangalore."))


def test_rejects_when_min_yoe_exceeds_user_plus_buffer(monkeypatch):
    monkeypatch.setattr(config, "USER_YOE", 4.5)
    monkeypatch.setattr(config, "YOE_BUFFER", 3.0)
    # cap = 7.5
    assert not experience_filter.passes_yoe_filter(_post("Requires 10+ years of experience."))
    assert not experience_filter.passes_yoe_filter(_post("Minimum 8 years experience."))
    assert not experience_filter.passes_yoe_filter(_post("8-12 years experience required."))


def test_accepts_when_min_yoe_within_buffer(monkeypatch):
    monkeypatch.setattr(config, "USER_YOE", 4.5)
    monkeypatch.setattr(config, "YOE_BUFFER", 3.0)
    # cap = 7.5, min = 5 → pass; min = 7 → pass; min = 7.5 → pass (== is allowed)
    assert experience_filter.passes_yoe_filter(_post("5-8 years of experience required."))
    assert experience_filter.passes_yoe_filter(_post("At least 7 years of hands-on experience."))


def test_junior_signal_rescues_borderline_post(monkeypatch):
    monkeypatch.setattr(config, "USER_YOE", 4.5)
    monkeypatch.setattr(config, "YOE_BUFFER", 3.0)
    # Explicit junior framing should pass even if a senior pattern also shows.
    jd = "Junior role. 0-2 years of experience. Team has 15 years of combined experience."
    assert experience_filter.passes_yoe_filter(_post(jd))


def test_filter_by_yoe_counts_rejections(monkeypatch):
    monkeypatch.setattr(config, "USER_YOE", 4.5)
    monkeypatch.setattr(config, "YOE_BUFFER", 3.0)
    posts = [
        _post("Great role, no year info."),
        _post("10+ years required."),
        _post("5-8 years experience."),
        _post("Minimum 12 years of leadership experience."),
    ]
    kept, rejected = experience_filter.filter_by_yoe(posts)
    assert len(kept) == 2
    assert rejected == 2


def test_filter_by_yoe_preserves_order(monkeypatch):
    monkeypatch.setattr(config, "USER_YOE", 4.5)
    monkeypatch.setattr(config, "YOE_BUFFER", 3.0)
    posts = [
        {"id": 1, "jd_text": "No years mentioned"},
        {"id": 2, "jd_text": "10+ years"},
        {"id": 3, "jd_text": "3-5 years of experience"},
    ]
    kept, _ = experience_filter.filter_by_yoe(posts)
    assert [p["id"] for p in kept] == [1, 3]


def test_falls_back_to_text_when_jd_text_missing():
    """Posts from sources that don't extract JD yet should still work."""
    post = {"text": "Need 15+ years of experience in consulting."}
    assert not experience_filter.passes_yoe_filter(post)
