"""Memory system implementation with LangChain integration."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from ..config.config import MemoryConfig

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Entry in the memory buffer."""
    role: str
    content: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class Memory(ConversationBufferMemory):
    """Memory system for storing and retrieving task examples."""

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        embeddings: Optional[Embeddings] = None,
        vector_store: Optional[VectorStore] = None,
        return_messages: bool = True,
        output_key: Optional[str] = None,
        input_key: Optional[str] = None,
        memory_key: str = "history",
    ):
        """Initialize the memory system.
        
        Args:
            config: Memory configuration
            embeddings: Embeddings model for text
            vector_store: Vector store for similarity search
            return_messages: Whether to return messages or strings
            output_key: Key to store outputs under
            input_key: Key to store inputs under
            memory_key: Key to store memory under
        """
        super().__init__(
            return_messages=return_messages,
            output_key=output_key,
            input_key=input_key,
            memory_key=memory_key
        )
        self._config = config or MemoryConfig()
        self._embeddings = embeddings
        self._vector_store = vector_store
        self._retriever = None
        if vector_store:
            self._retriever = vector_store.as_retriever(
                search_kwargs={
                    "k": 5,
                    "score_threshold": self._config.similarity_threshold
                }
            )
        self._buffer: List[MemoryEntry] = []
        self._summaries: List[str] = []
        self._summary_prompt = None
        self._init_prompts()

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Get the embeddings model."""
        return self._embeddings

    @property
    def vector_store(self) -> Optional[VectorStore]:
        """Get the vector store."""
        return self._vector_store

    @property
    def retriever(self) -> Optional[BaseRetriever]:
        """Get the retriever."""
        return self._retriever

    @property
    def buffer(self) -> List[MemoryEntry]:
        """Get the memory buffer."""
        return self._buffer

    @property
    def summaries(self) -> List[str]:
        """Get the summaries."""
        return self._summaries

    @property
    def summary_prompt(self) -> ChatPromptTemplate:
        """Get the summary prompt template."""
        return self._summary_prompt

    def _init_prompts(self) -> None:
        """Initialize prompt templates."""
        self._summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the following conversation:"),
            ("human", "{conversation}")
        ])

    def add_example(self, example: Dict[str, Any]) -> None:
        """Add a new example to memory.
        
        Args:
            example: Example to add
        """
        if len(self._buffer) >= self._config.max_examples:
            self._buffer.pop(0)
        
        entry = MemoryEntry(
            role=example.get("role", "user"),
            content=example.get("content", ""),
            timestamp=datetime.now().isoformat(),
            metadata=example.get("metadata", {})
        )
        self._buffer.append(entry)

    def get_relevant_examples(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get relevant examples for a query.
        
        Args:
            query: Query to find relevant examples for
            k: Number of examples to return
            
        Returns:
            List of relevant examples
        """
        if not self._retriever:
            return []
            
        docs = self._retriever.get_relevant_documents(query)
        return [doc.metadata for doc in docs]

    def _summarize(self, conversation: str) -> str:
        """Summarize a conversation.
        
        Args:
            conversation: Conversation to summarize
            
        Returns:
            Summary of the conversation
        """
        if not self._embeddings:
            return conversation
            
        chain = self._summary_prompt | self._embeddings | JsonOutputParser()
        result = chain.invoke({"conversation": conversation})
        return result.get("summary", conversation)

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load memory variables.
        
        Args:
            inputs: Input variables
            
        Returns:
            Memory variables
        """
        # First get the standard memory variables from parent class
        memory_vars = super().load_memory_variables(inputs)
        
        # Add our custom memory variables
        memory_vars.update({
            "buffer": self._buffer,
            "summaries": self._summaries
        })
        
        return memory_vars

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Save context to memory.
        
        Args:
            inputs: Input variables
            outputs: Output variables
        """
        # First save to parent class memory
        super().save_context(inputs, outputs)
        
        # Then save to our custom memory
        self.add_example({
            "role": "user",
            "content": str(inputs),
            "metadata": {"type": "input"}
        })
        self.add_example({
            "role": "assistant",
            "content": str(outputs),
            "metadata": {"type": "output"}
        })

    def clear(self) -> None:
        """Clear memory."""
        super().clear()
        self._buffer = []
        self._summaries = []

    def get_recent_actions(
        self,
        task_id: Optional[str] = None,
        max_actions: int = 5
    ) -> List[Dict[str, Any]]:
        """Get recent actions from memory."""
        actions = []
        for entry in reversed(self._buffer):
            if entry.role == "action":
                if task_id is None or entry.metadata.get("task_id") == task_id:
                    try:
                        action_data = json.loads(entry.content)
                        actions.append(action_data)
                        if len(actions) >= max_actions:
                            break
                    except json.JSONDecodeError:
                        continue
        return actions

    def get_langchain_memory(self) -> BaseMemory:
        """Get the underlying LangChain memory object."""
        return self

    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variables.
        
        Returns:
            List of memory variable names
        """
        return ["history", "examples", "context"]
