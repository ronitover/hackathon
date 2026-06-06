import memory
from rag_backend import build_retrieval_query


def test_build_retrieval_query_enriches_followup():
    history = [{"role": "user", "content": "What is the company leave policy?"}]
    enriched = build_retrieval_query("How many days?", history)
    assert "leave policy" in enriched.lower()
    assert "How many days?" in enriched


def test_build_retrieval_query_uses_assistant_context():
    history = [
        {"role": "user", "content": "What HR roles exist?"},
        {"role": "assistant", "content": "The document defines Manager, Analyst, and Coordinator roles."},
    ]
    enriched = build_retrieval_query("What do they do?", history)
    assert "Manager" in enriched or "roles" in enriched.lower()


def test_build_retrieval_query_keeps_standalone_question():
    history = [{"role": "user", "content": "What is the leave policy?"}]
    query = "Summarize the onboarding process for new engineers in detail"
    assert build_retrieval_query(query, history) == query


def test_memory_roundtrip_with_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_DB_PATH", tmp_path / "test_memory.db")
    session = memory.new_session_id()
    memory.ensure_session(session)
    memory.add_message(session, "user", "Hello")
    memory.add_message(
        session,
        "assistant",
        "Hi there",
        grounded=True,
        sources=[{"source": "doc.pdf", "page": 0, "excerpt": "sample text"}],
    )
    messages = memory.get_messages(session)
    assert len(messages) == 2
    assert messages[1]["grounded"] is True
    assert messages[1]["sources"][0]["excerpt"] == "sample text"
    assert memory.get_history_for_rag(session)[0]["role"] == "user"
    memory.clear_session(session)
    assert memory.get_messages(session) == []
