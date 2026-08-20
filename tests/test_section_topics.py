from src.section_topics import CANONICAL_SECTIONS, infer_topic, matches_topic, normalize_section


def test_normalize_section_strips_prefix_and_punctuation():
    assert normalize_section("SECTION II – RISK FACTORS") == "RISK FACTORS"
    assert normalize_section("SECTION II: RISK FACTORS") == "RISK FACTORS"
    assert normalize_section("SECTION II - RISK FACTORS") == "RISK FACTORS"


def test_normalize_section_handles_none_and_empty():
    assert normalize_section(None) is None
    assert normalize_section("") is None
    assert normalize_section("   ") is None


def test_normalize_section_collapses_hyphenated_words():
    # Real corpus example: "FORWARD-LOOKING STATEMENTS" -> hyphen dropped,
    # not replaced with a space.
    assert normalize_section("FORWARD-LOOKING STATEMENTS") == "FORWARDLOOKING STATEMENTS"


def test_matches_topic_handles_issue_offer_synonyms():
    # Different merchant-banker templates use "Issue" and "Offer"
    # interchangeably for the same section.
    assert matches_topic("SECTION III – OBJECTS OF THE ISSUE", "OBJECTS OF THE ISSUE") is True
    assert matches_topic("OBJECTS OF THE OFFER", "OBJECTS OF THE ISSUE") is True


def test_matches_topic_rejects_unrelated_section():
    assert matches_topic("CAPITAL STRUCTURE", "RISK FACTORS") is False


def test_matches_topic_handles_none_section():
    assert matches_topic(None, "RISK FACTORS") is False


def test_matches_topic_unknown_topic_matches_nothing():
    assert matches_topic("RISK FACTORS", "NOT A REAL TOPIC") is False


def test_infer_topic_recognizes_risk_factors():
    assert infer_topic("What are the risk factors for Aastha Spintex Limited?") == "RISK FACTORS"


def test_infer_topic_recognizes_objects_of_the_issue():
    assert infer_topic("What are the objects of the issue for Advit Jewels?") == "OBJECTS OF THE ISSUE"
    assert infer_topic("How will the offer proceeds be used?") == "OBJECTS OF THE ISSUE"


def test_infer_topic_returns_none_for_unrecognized_question():
    assert infer_topic("What raw materials does the company depend on?") is None


def test_infer_topic_recognizes_issue_mechanics():
    # These facts are already isolated as their own correctly section-tagged
    # table chunks by src/table_extractor.py, but without a trigger phrase
    # here infer_topic() silently fell back to plain DEFAULT_TOP_K search
    # instead of the section-scoped BROAD_TOP_K a recognized topic gets.
    assert infer_topic("What is the price band and lot size for Acme?") == "THE ISSUE"
    assert infer_topic("What is the QIB quota for Acme's IPO?") == "ISSUE STRUCTURE"
    assert infer_topic("Who is the registrar to the issue for Acme?") == "GENERAL INFORMATION"


def test_every_topic_trigger_maps_to_a_real_canonical_section():
    # infer_topic() and matches_topic() must stay in sync -- a topic that
    # can be inferred but never matches a real section would silently
    # produce zero results.
    from src.section_topics import _TOPIC_TRIGGERS

    for topic in _TOPIC_TRIGGERS:
        assert topic in CANONICAL_SECTIONS, f"{topic!r} has triggers but no canonical section entry"
