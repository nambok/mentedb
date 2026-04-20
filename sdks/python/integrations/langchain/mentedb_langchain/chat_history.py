"""LangChain chat history backed by MenteDB."""
from typing import List, Optional

from mentedb import MenteDB


class MenteDBChatHistory:
    """Stores chat message history in MenteDB with persistent recall.

    Each message is stored as an episodic memory, enabling semantic
    search across conversation history and cross-session continuity.

    Usage:
        from mentedb_langchain import MenteDBChatHistory

        history = MenteDBChatHistory(
            session_id="session-123",
            data_dir="./agent-memory",
        )

        history.add_user_message("What database should I use?")
        history.add_ai_message("I recommend PostgreSQL.")

        messages = history.messages
    """

    def __init__(self, session_id: str, data_dir: str = "./mentedb-data",
                 agent_id: Optional[str] = None):
        self.session_id = session_id
        self.data_dir = data_dir
        self.agent_id = agent_id
        self._db = MenteDB(data_dir)
        self._messages: list = []

    @property
    def messages(self) -> list:
        return self._messages

    def add_user_message(self, message: str) -> None:
        self._messages.append({"role": "user", "content": message})
        self._db.store(
            f"[{self.session_id}] User: {message}",
            memory_type="episodic",
            agent_id=self.agent_id,
            tags=["chat_history", self.session_id],
        )

    def add_ai_message(self, message: str) -> None:
        self._messages.append({"role": "assistant", "content": message})
        self._db.store(
            f"[{self.session_id}] Assistant: {message}",
            memory_type="episodic",
            agent_id=self.agent_id,
            tags=["chat_history", self.session_id],
        )

    def clear(self) -> None:
        self._messages.clear()
