"""Query builder: role keyword extraction + 5-tier ladder, no city filter."""
from __future__ import annotations

from src import query_builder


def _track(roles=None, include_remote=True, active=True):
    return {
        "track": "t",
        "roles": roles if roles is not None else ["Growth Marketing Manager"],
        # cities may still be present in prefs from older runs; the builder
        # must ignore them entirely.
        "cities": ["Bangalore", "Gurgaon"],
        "include_remote": include_remote,
        "active": active,
    }


# ---------- role keyword extraction ----------

def test_role_specific_drops_seniority_and_generic_suffix():
    assert query_builder._role_specific("Senior Category Manager") == "Category"
    assert query_builder._role_specific("Sr. Product Manager") == "Product"
    assert query_builder._role_specific("Junior Data Analyst") == "Data Analyst"


def test_role_specific_drops_only_manager_not_other_titles():
    # Director, Lead, Consultant, Head carry role signal — keep them.
    assert query_builder._role_specific("Account Director") == "Account Director"
    assert query_builder._role_specific("Automation Lead") == "Automation Lead"
    assert query_builder._role_specific("Marketing Consultant") == "Marketing Consultant"
    assert query_builder._role_specific("Head of Growth") == "Head of Growth"


def test_role_specific_preserves_compound_titles():
    assert query_builder._role_specific("Growth Marketing Manager") == "Growth Marketing"
    assert query_builder._role_specific("Performance Marketing Manager") == "Performance Marketing"
    assert query_builder._role_specific("AI Operations Manager") == "AI Operations"
    assert query_builder._role_specific("Digital Transformation Manager") == "Digital Transformation"


def test_role_specific_handles_single_word_after_strip():
    # Stripping "Manager" would leave empty → fall back to original
    assert query_builder._role_specific("Manager") == "Manager"
    # "Brand Manager" → "Brand"
    assert query_builder._role_specific("Brand Manager") == "Brand"


def test_role_broad_picks_first_meaningful_token():
    assert query_builder._role_broad("Growth Marketing Manager") == "Growth"
    assert query_builder._role_broad("Performance Marketing Manager") == "Performance"
    assert query_builder._role_broad("Head of Growth") == "Growth"
    assert query_builder._role_broad("Head of AI") == "AI"
    assert query_builder._role_broad("Account Director") == "Account"
    assert query_builder._role_broad("Automation Lead") == "Automation"


def test_role_broad_keeps_acronyms():
    assert query_builder._role_broad("AI Operations Manager") == "AI"
    assert query_builder._role_broad("AI Transformation Lead") == "AI"


def test_role_broad_fallback_when_all_tokens_are_bad():
    # "Chief of Staff" → Chief/of/Staff are all bad-broads → fallback to full specific
    assert query_builder._role_broad("Chief of Staff") == "Chief of Staff"


def test_role_broad_handles_empty_input():
    assert query_builder._role_broad("") == ""


# ---------- query ladder ----------

def test_empty_roles_returns_empty():
    assert query_builder.build_queries_for_track(_track(roles=[])) == []


def test_specific_quoted_query_is_produced():
    qs = query_builder.build_queries_for_track(_track(roles=["Growth Marketing Manager"]))
    # T1: #hiring "Growth Marketing"
    assert any('#hiring "Growth Marketing"' == q for q in qs)


def test_city_is_never_injected():
    qs = query_builder.build_queries_for_track(_track())
    assert not any("Bangalore" in q or "Gurgaon" in q for q in qs)


def test_india_scope_query_present():
    qs = query_builder.build_queries_for_track(_track(roles=["Growth Marketing Manager"]))
    # T2: India scope
    assert any("India" in q for q in qs)


def test_we_are_hiring_non_hashtag_query_present():
    qs = query_builder.build_queries_for_track(_track(roles=["Growth Marketing Manager"]))
    # T3
    assert any('"we are hiring"' in q.lower() for q in qs)


def test_remote_queries_added_when_enabled():
    qs = query_builder.build_queries_for_track(_track(include_remote=True))
    assert any("remote" in q for q in qs)


def test_remote_queries_skipped_when_disabled():
    qs = query_builder.build_queries_for_track(_track(include_remote=False))
    assert not any("remote" in q for q in qs)


def test_hashtag_queries_present():
    qs = query_builder.build_queries_for_track(_track())
    assert any("#hiring" in q for q in qs)


def test_broad_safety_net_query_present():
    qs = query_builder.build_queries_for_track(_track(roles=["Growth Marketing Manager"]))
    # T5: #hiring Growth India
    assert any(q == "#hiring Growth India" for q in qs)


def test_query_cap_respected():
    many = _track(roles=[f"Role {i} Manager" for i in range(12)])
    qs = query_builder.build_queries_for_track(many)
    from src import config
    assert len(qs) <= config.QUERIES_PER_TRACK


def test_build_all_skips_inactive():
    prefs = {
        "tracks": [
            {**_track(), "track": "active_track", "active": True},
            {**_track(), "track": "inactive_track", "active": False},
        ]
    }
    all_qs = query_builder.build_all_queries(prefs)
    assert "active_track" in all_qs
    assert "inactive_track" not in all_qs


def test_no_duplicate_queries():
    qs = query_builder.build_queries_for_track(_track())
    assert len(qs) == len(set(qs))


def test_multi_role_track_covers_all_top_specifics():
    """With 4 different roles, all 4 should produce T1 queries."""
    roles = [
        "Growth Marketing Manager",
        "Performance Marketing Manager",
        "Brand Manager",
        "Product Marketing Manager",
    ]
    qs = query_builder.build_queries_for_track(_track(roles=roles))
    assert any('"Growth Marketing"' in q for q in qs)
    assert any('"Performance Marketing"' in q for q in qs)
    assert any('"Brand"' in q for q in qs)
    assert any('"Product Marketing"' in q for q in qs)


def test_chief_of_staff_produces_reasonable_queries():
    """Edge case: every token is in _BAD_BROADS; should still produce queries."""
    qs = query_builder.build_queries_for_track(_track(roles=["Chief of Staff"]))
    assert any('"Chief of Staff"' in q for q in qs)
    # T5 fallback to full specific when no single broad keyword qualifies
    assert any("Chief of Staff" in q for q in qs)


# ---------- T6 synonym catchment ----------


def test_synonyms_for_known_phrase():
    syns = query_builder._synonyms_for("Growth Marketing")
    assert "performance marketing" in syns
    assert "growth hacker" in syns


def test_synonyms_for_unknown_phrase_returns_empty():
    assert query_builder._synonyms_for("Xylophone Wrangler") == []


def test_synonyms_fall_back_to_broad_token(monkeypatch):
    """If the full specific has no entry but the broad does, we should use that."""
    # "Data Analytics" has its own key, but "data" → data-science variants
    # would fire if the phrase key were missing. Prove the fallback by
    # querying a made-up specific whose broad token IS keyed.
    monkeypatch.setitem(query_builder._ROLE_SYNONYMS, "growth", ["performance"])
    syns = query_builder._synonyms_for("Growth Widget")
    assert "performance" in syns


def test_synonym_tier_emits_quoted_queries_for_known_roles():
    """Running the full ladder for a keyed role should produce T6 synonym queries."""
    # Raise the cap so T6 isn't pruned
    from src import config
    original = config.QUERIES_PER_TRACK
    try:
        config.QUERIES_PER_TRACK = 20
        qs = query_builder.build_queries_for_track(_track(roles=["Growth Marketing Manager"]))
    finally:
        config.QUERIES_PER_TRACK = original

    joined = "\n".join(qs)
    # At least one #hiring "<synonym>" should have landed.
    assert '#hiring "performance marketing"' in joined or '#hiring "growth hacker"' in joined


def test_synonym_tier_silent_for_unknown_roles():
    """A role with no synonym mapping shouldn't crash the ladder."""
    qs = query_builder.build_queries_for_track(_track(roles=["Underwater Basket Weaver"]))
    # T1 still fires
    assert any('"Underwater Basket Weaver"' in q for q in qs)
