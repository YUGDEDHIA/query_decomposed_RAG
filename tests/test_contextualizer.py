from __future__ import annotations

from src.contextualizer import MAX_HISTORY_TURNS, Turn, _format_history, _truncate, contextualize


def test_truncate_leaves_short_text_unchanged():
    assert _truncate("short answer") == "short answer"


def test_truncate_cuts_long_text_with_ellipsis():
    text = "x" * 600
    result = _truncate(text, max_chars=500)
    assert len(result) == 503  # 500 chars + "..."
    assert result.endswith("...")


def test_truncate_exact_boundary_is_unchanged():
    text = "x" * 500
    assert _truncate(text, max_chars=500) == text


def test_format_history_includes_user_and_assistant_lines():
    history = [Turn(question="What are Acme's risk factors?", answer="Acme faces several risks...")]
    formatted = _format_history(history)
    assert "User: What are Acme's risk factors?" in formatted
    assert "Assistant: Acme faces several risks..." in formatted


def test_format_history_truncates_long_answers():
    history = [Turn(question="Q", answer="x" * 600)]
    formatted = _format_history(history)
    assert "x" * 600 not in formatted
    assert "..." in formatted


def test_format_history_keeps_only_last_max_turns():
    history = [Turn(question=f"Q{i}", answer=f"A{i}") for i in range(MAX_HISTORY_TURNS + 3)]
    formatted = _format_history(history)
    for i in range(MAX_HISTORY_TURNS + 3 - MAX_HISTORY_TURNS):
        assert f"Q{i}" not in formatted
    for i in range(MAX_HISTORY_TURNS + 3 - MAX_HISTORY_TURNS, MAX_HISTORY_TURNS + 3):
        assert f"Q{i}" in formatted


def test_contextualize_returns_question_unchanged_when_history_empty(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("should not call ollama.chat when history is empty")

    monkeypatch.setattr("ollama.chat", _fail)

    assert contextualize("What are Acme's risk factors?", []) == "What are Acme's risk factors?"
