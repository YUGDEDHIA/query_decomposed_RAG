from __future__ import annotations

from src.answerer import SubAnswer
from src.contextualizer import Turn
from src.pipeline import answer_question
from src.synthesizer import FinalAnswer


class _FakeRetriever:
    def __init__(self, known_company="Acme", fuzzy_match=None):
        # Default non-None so unrelated tests don't accidentally trigger
        # the on-demand ingestion path just by using this fake.
        self._known_company = known_company
        self._fuzzy_match = fuzzy_match
        self.retrieve_calls = []

    def infer_company(self, query):
        return self._known_company

    def fuzzy_match_company(self, name):
        return self._fuzzy_match

    def retrieve(self, query, **kwargs):
        self.retrieve_calls.append(kwargs)
        return [{"text": "chunk"}]


def test_casual_intent_skips_retrieval_and_planner(monkeypatch):
    decompose_calls = []
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: "casual")
    monkeypatch.setattr("src.pipeline.casual_reply", lambda q: "Hello! Ask me about an RHP filing.")
    monkeypatch.setattr("src.pipeline.decompose", lambda q: decompose_calls.append(q))

    result = answer_question("Hii", _FakeRetriever())

    assert result.intent == "casual"
    assert result.answer == "Hello! Ask me about an RHP filing."
    assert result.sub_answers == []
    assert result.sources == []
    assert decompose_calls == []  # the research pipeline must never run for casual intent


def test_rhp_query_intent_runs_full_pipeline(monkeypatch):
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: "rhp_query")
    monkeypatch.setattr("src.pipeline.decompose", lambda q: [q])
    monkeypatch.setattr(
        "src.pipeline.answer",
        lambda sq, chunks: SubAnswer(question=sq, answer="Some risks.", grounded=True, sources=[]),
    )
    monkeypatch.setattr(
        "src.pipeline.synthesize",
        lambda q, subs: FinalAnswer(
            question=q, answer=subs[0].answer, fully_grounded=True, sub_answers=subs, sources=[], intent="rhp_query",
        ),
    )

    result = answer_question("What are the risk factors for Acme?", _FakeRetriever())

    assert result.intent == "rhp_query"
    assert result.answer == "Some risks."


def test_gmp_intent_skips_retrieval_and_planner(monkeypatch):
    decompose_calls = []
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: "gmp_query")
    monkeypatch.setattr("src.pipeline.answer_gmp_question", lambda q: "GMP for Acme is ₹50.")
    monkeypatch.setattr("src.pipeline.decompose", lambda q: decompose_calls.append(q))

    result = answer_question("what's the GMP for Acme?", _FakeRetriever())

    assert result.intent == "gmp_query"
    assert result.answer == "GMP for Acme is ₹50."
    assert result.sub_answers == []
    assert result.sources == []
    assert decompose_calls == []


def test_subscription_intent_skips_retrieval_and_planner(monkeypatch):
    decompose_calls = []
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: "subscription_query")
    monkeypatch.setattr("src.pipeline.answer_subscription_question", lambda q: "Acme is subscribed 2.5x.")
    monkeypatch.setattr("src.pipeline.decompose", lambda q: decompose_calls.append(q))

    result = answer_question("what's the subscription status for Acme?", _FakeRetriever())

    assert result.intent == "subscription_query"
    assert result.answer == "Acme is subscribed 2.5x."
    assert result.sub_answers == []
    assert result.sources == []
    assert decompose_calls == []


def test_listing_performance_intent_skips_retrieval_and_planner(monkeypatch):
    decompose_calls = []
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: "listing_performance_query")
    monkeypatch.setattr("src.pipeline.answer_listing_performance_question", lambda q: "Acme is up 23% since listing.")
    monkeypatch.setattr("src.pipeline.decompose", lambda q: decompose_calls.append(q))

    result = answer_question("how has Acme performed since listing?", _FakeRetriever())

    assert result.intent == "listing_performance_query"
    assert result.answer == "Acme is up 23% since listing."
    assert result.sub_answers == []
    assert result.sources == []
    assert decompose_calls == []


def test_cross_company_intent_skips_planner_and_uses_compare_companies(monkeypatch):
    decompose_calls = []
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: "cross_company_query")
    monkeypatch.setattr(
        "src.pipeline.compare_companies",
        lambda q, retriever: ("Acme has the strongest financials.", [{"company": "Acme"}]),
    )
    monkeypatch.setattr("src.pipeline.decompose", lambda q: decompose_calls.append(q))

    result = answer_question("which IPOs have good financials?", _FakeRetriever())

    assert result.intent == "cross_company_query"
    assert result.answer == "Acme has the strongest financials."
    assert result.fully_grounded is True
    assert result.sources == [{"company": "Acme"}]
    assert decompose_calls == []


def test_cross_company_intent_not_grounded_when_no_sources_found(monkeypatch):
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: "cross_company_query")
    monkeypatch.setattr("src.pipeline.compare_companies", lambda q, retriever: ("Nothing found.", []))

    result = answer_question("which IPOs have good financials?", _FakeRetriever())

    assert result.fully_grounded is False


def test_rhp_query_skips_on_demand_ingest_when_company_already_known(monkeypatch):
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: "rhp_query")
    monkeypatch.setattr("src.pipeline.decompose", lambda q: [q])
    monkeypatch.setattr("src.pipeline.answer", lambda sq, chunks: SubAnswer(question=sq, answer="A", grounded=True, sources=[]))
    monkeypatch.setattr("src.pipeline.synthesize", lambda q, subs: FinalAnswer(question=q, answer="A", fully_grounded=True, sub_answers=subs, sources=[]))

    extract_calls = []
    monkeypatch.setattr("src.pipeline.extract_company_name", lambda q: extract_calls.append(q))
    monkeypatch.setattr("src.pipeline.ingest_company_on_demand", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not be called")))

    answer_question("What are the risk factors for Acme?", _FakeRetriever(known_company="Acme"))

    assert extract_calls == []  # company already known -- no need to even try extraction


def test_rhp_query_triggers_on_demand_ingest_for_unknown_company(monkeypatch):
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: "rhp_query")
    monkeypatch.setattr("src.pipeline.decompose", lambda q: [q])
    monkeypatch.setattr("src.pipeline.answer", lambda sq, chunks: SubAnswer(question=sq, answer="A", grounded=True, sources=[]))
    monkeypatch.setattr("src.pipeline.synthesize", lambda q, subs: FinalAnswer(question=q, answer="A", fully_grounded=True, sub_answers=subs, sources=[]))
    monkeypatch.setattr("src.pipeline.extract_company_name", lambda q: "New Corp")

    ingest_calls = []
    monkeypatch.setattr(
        "src.pipeline.ingest_company_on_demand",
        lambda candidate, retriever, on_progress=None: ingest_calls.append(candidate) or "New Corp Limited",
    )

    answer_question("What are the risk factors for New Corp?", _FakeRetriever(known_company=None))

    assert ingest_calls == ["New Corp"]


def test_rhp_query_uses_fuzzy_match_instead_of_ingest_for_misspelled_company(monkeypatch):
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: "rhp_query")
    monkeypatch.setattr("src.pipeline.decompose", lambda q: [q])
    monkeypatch.setattr("src.pipeline.answer", lambda sq, chunks: SubAnswer(question=sq, answer="A", grounded=True, sources=[]))
    monkeypatch.setattr("src.pipeline.synthesize", lambda q, subs: FinalAnswer(question=q, answer="A", fully_grounded=True, sub_answers=subs, sources=[]))
    monkeypatch.setattr("src.pipeline.extract_company_name", lambda q: "Aastha Spintx Limited")
    monkeypatch.setattr(
        "src.pipeline.ingest_company_on_demand",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not be called -- fuzzy match should resolve it first")),
    )

    retriever = _FakeRetriever(known_company=None, fuzzy_match="Aastha Spintex Limited")
    answer_question("What are the risk factors for Aastha Spintx Limited?", retriever)

    # The corrected, indexed name -- not the misspelled one -- must be what
    # scopes retrieval, since the sub-question's own text is still misspelled.
    assert retriever.retrieve_calls == [{"company": "Aastha Spintex Limited"}]


def test_rhp_query_falls_back_to_ingest_when_no_fuzzy_match(monkeypatch):
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: "rhp_query")
    monkeypatch.setattr("src.pipeline.decompose", lambda q: [q])
    monkeypatch.setattr("src.pipeline.answer", lambda sq, chunks: SubAnswer(question=sq, answer="A", grounded=True, sources=[]))
    monkeypatch.setattr("src.pipeline.synthesize", lambda q, subs: FinalAnswer(question=q, answer="A", fully_grounded=True, sub_answers=subs, sources=[]))
    monkeypatch.setattr("src.pipeline.extract_company_name", lambda q: "New Corp")
    monkeypatch.setattr(
        "src.pipeline.ingest_company_on_demand",
        lambda candidate, retriever, on_progress=None: "New Corp Limited",
    )

    retriever = _FakeRetriever(known_company=None, fuzzy_match=None)
    answer_question("What are the risk factors for New Corp?", retriever)

    assert retriever.retrieve_calls == [{"company": "New Corp Limited"}]


def test_rhp_query_does_not_ingest_when_no_candidate_name_extracted(monkeypatch):
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: "rhp_query")
    monkeypatch.setattr("src.pipeline.decompose", lambda q: [q])
    monkeypatch.setattr("src.pipeline.answer", lambda sq, chunks: SubAnswer(question=sq, answer="A", grounded=True, sources=[]))
    monkeypatch.setattr("src.pipeline.synthesize", lambda q, subs: FinalAnswer(question=q, answer="A", fully_grounded=True, sub_answers=subs, sources=[]))
    monkeypatch.setattr("src.pipeline.extract_company_name", lambda q: None)
    monkeypatch.setattr("src.pipeline.ingest_company_on_demand", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not be called")))

    # Should not raise -- no candidate name means no ingestion attempt.
    answer_question("What's a good IPO strategy in general?", _FakeRetriever(known_company=None))


def test_progress_callback_reports_classification_before_routing(monkeypatch):
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: "casual")
    monkeypatch.setattr("src.pipeline.casual_reply", lambda q: "Hi there!")

    progress_messages = []
    answer_question("Hii", _FakeRetriever(), on_progress=progress_messages.append)

    assert progress_messages[0] == "Classifying question..."
    assert "Responding..." in progress_messages


def test_history_none_never_calls_contextualize(monkeypatch):
    # Every pre-existing call site (chat.py's first turn, server.py's first
    # request in a session, every test above) omits history entirely --
    # this must be a complete no-op, not just an empty-list short-circuit.
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: "casual")
    monkeypatch.setattr("src.pipeline.casual_reply", lambda q: "Hi there!")
    monkeypatch.setattr(
        "src.pipeline.contextualize",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not be called when history is None")),
    )

    answer_question("Hii", _FakeRetriever())


def test_history_empty_list_never_calls_contextualize(monkeypatch):
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: "casual")
    monkeypatch.setattr("src.pipeline.casual_reply", lambda q: "Hi there!")
    monkeypatch.setattr(
        "src.pipeline.contextualize",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not be called when history is empty")),
    )

    answer_question("Hii", _FakeRetriever(), history=[])


def test_history_non_empty_resolves_question_before_classification(monkeypatch):
    classify_calls = []
    monkeypatch.setattr("src.pipeline.contextualize", lambda q, h: "What is Acme Ltd's GMP?")
    monkeypatch.setattr("src.pipeline.classify_intent", lambda q: classify_calls.append(q) or "gmp_query")
    monkeypatch.setattr("src.pipeline.answer_gmp_question", lambda q: "Acme Ltd's GMP is ₹50.")

    history = [Turn(question="What are Acme Ltd's risk factors?", answer="Acme Ltd faces several risks...")]
    result = answer_question("what about its GMP?", _FakeRetriever(), history=history)

    # classify_intent (and, downstream, answer_gmp_question) must see the
    # resolved standalone question, not the raw pronoun-laden one.
    assert classify_calls == ["What is Acme Ltd's GMP?"]
    assert result.answer == "Acme Ltd's GMP is ₹50."
