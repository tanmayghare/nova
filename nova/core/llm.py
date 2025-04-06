"""Language model integration."""

from typing import Any, Dict, List, Optional, Union, Protocol, cast

from langchain_core.language_models.chat_models import BaseChatModel

from .llama import LlamaModel


class LanguageModel(Protocol):
    """Protocol for language models."""
    
    async def generate_plan(self, task: str, context: str) -> List[Dict[str, Any]]:
        """Generate a plan for executing a task."""
        ...
    
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results."""
        ...


class LangChainAdapter:
    """Adapter for LangChain models to implement the LanguageModel protocol."""
    
    def __init__(self, model: BaseChatModel):
        """Initialize with a LangChain model."""
        self.model = model
    
    async def generate_plan(self, task: str, context: str) -> List[Dict[str, Any]]:
        """Generate a plan for executing a task."""
        # TODO: Implement actual plan generation for LangChain models
        return [
            {"type": "tool", "tool": "mock", "input": {"test": "data"}},
            {"type": "browser", "action": {"type": "navigate", "url": "https://example.com"}},
        ]
    
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results."""
        # TODO: Implement actual response generation for LangChain models
        return "Final response"


class LLM:
    """Wrapper for language models.
    
    This class provides a unified interface for different language models,
    including both LangChain models and Llama.
    """

    def __init__(
        self,
        model: Union[BaseChatModel, LlamaModel],
        model_type: str = "langchain",
    ):
        """Initialize with a language model.
        
        Args:
            model: The language model instance
            model_type: Type of model ("langchain" or "llama")
        """
        self.model: LanguageModel
        if model_type == "langchain":
            self.model = LangChainAdapter(cast(BaseChatModel, model))
        else:
            self.model = cast(LanguageModel, model)

    async def generate_plan(self, task: str, context: str) -> List[Dict[str, Any]]:
        """Generate a plan for executing a task."""
        return await self.model.generate_plan(task, context)

    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results."""
        return await self.model.generate_response(task, plan, context) 