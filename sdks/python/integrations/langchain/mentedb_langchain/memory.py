"""LangChain memory backed by MenteDB."""
from typing import Any, Dict, List, Optional

from mentedb import MenteDB


class MenteDBMemory:
    """LangChain compatible memory that stores conversation context in MenteDB.

    Usage with LangChain:
        from mentedb_langchain import MenteDBMemory

        memory = MenteDBMemory(data_dir="./agent-memory", agent_id="my-agent")

        memory.save_context(
            inputs={"input": "What database should I use?"},
            outputs={"output": "I recommend PostgreSQL for your use case."}
        )

        context = memory.load_memory_variables({"input": "Tell me more about that database"})
    """

    memory_key: str = "history"

    def __init__(self, data_dir: str = "./mentedb-data", agent_id: Optional[str] = None,
                 token_budget: int = 4096, return_messages: bool = False):
        self.data_dir = data_dir
        self.agent_id = agent_id
        self.token_budget = token_budget
        self.return_messages = return_messages
        self._db = MenteDB(data_dir)

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load relevant memories based on the current input."""
        query = inputs.get("input", "")
        if not query:
            return {self.memory_key: ""}
        results = self._db.search_text(query, k=10)
        if not results:
            return {self.memory_key: ""}
        texts = []
        for r in results:
            mem = self._db.get_memory(r.id)
            if mem:
                texts.append(mem["content"])
        return {self.memory_key: "\n".join(texts)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save a conversation turn as episodic memory."""
        input_text = inputs.get("input", "")
        output_text = outputs.get("output", "")
        content = f"User: {input_text}\nAssistant: {output_text}"
        self._db.store(content, memory_type="episodic", agent_id=self.agent_id)

    def clear(self) -> None:
        """Clear all memories by closing and reopening."""
        self._db.close()
        self._db = MenteDB(self.data_dir)

