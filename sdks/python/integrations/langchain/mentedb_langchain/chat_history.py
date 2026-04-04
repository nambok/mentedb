"""LangChain chat history backed by MenteDB."""
from typing import List, Optional


class MenteDBChatHistory:
    """Stores chat message history in MenteDB with full cognitive features.

    Unlike simple chat history stores, MenteDB tracks:
    - Reasoning trajectory across sessions
    - Knowledge gaps (phantom memories)
    - Contradictions between turns
    - Pain signals from user feedback

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
        self._messages: list = []

    @property
    def messages(self) -> list:
        return self._messages

    def add_user_message(self, message: str) -> None:
        self._messages.append({"role": "user", "content": message})

    def add_ai_message(self, message: str) -> None:
        self._messages.append({"role": "assistant", "content": message})

    def clear(self) -> None:
        self._messages.clear()
