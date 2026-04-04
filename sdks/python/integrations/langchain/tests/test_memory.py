"""Tests for mentedb_langchain integration."""
from mentedb_langchain.memory import MenteDBMemory
from mentedb_langchain.retriever import MenteDBRetriever
from mentedb_langchain.chat_history import MenteDBChatHistory


class TestMenteDBMemory:
    def test_default_construction(self):
        memory = MenteDBMemory()
        assert memory.data_dir == "./mentedb-data"
        assert memory.agent_id is None
        assert memory.token_budget == 4096
        assert memory.return_messages is False

    def test_custom_construction(self):
        memory = MenteDBMemory(
            data_dir="./custom-dir",
            agent_id="agent-1",
            token_budget=8192,
            return_messages=True,
        )
        assert memory.data_dir == "./custom-dir"
        assert memory.agent_id == "agent-1"
        assert memory.token_budget == 8192
        assert memory.return_messages is True

    def test_memory_variables(self):
        memory = MenteDBMemory()
        assert memory.memory_variables == ["history"]

    def test_load_memory_variables(self):
        memory = MenteDBMemory()
        result = memory.load_memory_variables({"input": "test query"})
        assert "history" in result
        assert isinstance(result["history"], str)

    def test_save_context(self):
        memory = MenteDBMemory()
        memory.save_context(
            inputs={"input": "hello"},
            outputs={"output": "world"},
        )

    def test_clear(self):
        memory = MenteDBMemory()
        memory.clear()


class TestMenteDBRetriever:
    def test_default_construction(self):
        retriever = MenteDBRetriever()
        assert retriever.data_dir == "./mentedb-data"
        assert retriever.k == 10
        assert retriever.tags is None
        assert retriever.agent_id is None

    def test_custom_construction(self):
        retriever = MenteDBRetriever(
            data_dir="./custom-dir",
            k=20,
            tags=["backend"],
            agent_id="agent-1",
        )
        assert retriever.k == 20
        assert retriever.tags == ["backend"]

    def test_get_relevant_documents(self):
        retriever = MenteDBRetriever()
        docs = retriever.get_relevant_documents("test query")
        assert isinstance(docs, list)


class TestMenteDBChatHistory:
    def test_default_construction(self):
        history = MenteDBChatHistory(session_id="s1")
        assert history.session_id == "s1"
        assert history.data_dir == "./mentedb-data"
        assert history.agent_id is None

    def test_messages_empty(self):
        history = MenteDBChatHistory(session_id="s1")
        assert history.messages == []

    def test_add_messages(self):
        history = MenteDBChatHistory(session_id="s1")
        history.add_user_message("hello")
        history.add_ai_message("hi there")
        assert len(history.messages) == 2
        assert history.messages[0] == {"role": "user", "content": "hello"}
        assert history.messages[1] == {"role": "assistant", "content": "hi there"}

    def test_clear(self):
        history = MenteDBChatHistory(session_id="s1")
        history.add_user_message("hello")
        history.clear()
        assert history.messages == []
