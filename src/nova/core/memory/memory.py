"""Memory system implementation leveraging LangChain ConversationBufferMemory."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMemory

logger = logging.getLogger(__name__)

class Memory(ConversationBufferMemory):
    """Memory system focused on conversation history buffering.
    Inherits directly from ConversationBufferMemory, keeping it simple.
    """

    def __init__(
        self,
        llm: Optional[BaseMemory] = None,
        memory_key: str = "history",
        input_key: Optional[str] = None,
        output_key: Optional[str] = None,
        return_messages: bool = True,
        max_token_limit: Optional[int] = None
    ):
        """Initialize the memory system.
        
        Args:
            llm: Optional LLM for summarization (if parent class uses it).
            memory_key: Key to store memory under.
            input_key: Key for input variable.
            output_key: Key for output variable.
            return_messages: Whether to return messages or strings.
            max_token_limit: Optional token limit for buffer pruning.
        """
        super().__init__(
            llm=llm,
            memory_key=memory_key,
            input_key=input_key,
            output_key=output_key,
            return_messages=return_messages,
        )
        logger.debug(
            f"Initialized Memory (ConversationBufferMemory) with key '{memory_key}', "
            f"return_messages={return_messages}."
        )

    def get_langchain_memory(self) -> BaseMemory:
        """Get the underlying LangChain memory object."""
        return self
