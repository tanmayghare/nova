from __future__ import annotations

"""Language model integration."""

from typing import Any, Dict, List, Optional, Union, Protocol, cast

import logging
import json

from langchain_core.language_models.chat_models import BaseChatModel
from ollama import AsyncClient

from .llama import LlamaModel

logger = logging.getLogger(__name__)


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
    """Language model interface for Nova."""
    
    def __init__(self, model_name: str = "llama3.2:3b-instruct-q8_0") -> None:
        """Initialize the language model.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.client = AsyncClient()
        
    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate a response from the model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate (ignored for Ollama)
            
        Returns:
            Generated response
            
        Raises:
            RuntimeError: If model fails to generate response
        """
        try:
            response = await self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,  # Ollama uses num_predict instead of max_tokens
                }
            )
            
            if not response or not response.get("response"):
                raise RuntimeError("Model returned empty response")
                
            return response["response"]
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate response: {str(e)}")
            
    async def chat(self, messages: List[Dict[str, str]], max_tokens: int = 1000) -> str:
        """Generate a response from the model using chat format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum number of tokens to generate (ignored for Ollama)
            
        Returns:
            Generated response
            
        Raises:
            RuntimeError: If model fails to generate response
        """
        try:
            response = await self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "num_predict": max_tokens,  # Ollama uses num_predict instead of max_tokens
                }
            )
            
            if not response or not response.get("message"):
                raise RuntimeError("Model returned empty response")
                
            return response["message"]["content"]
            
        except Exception as e:
            logger.error(f"LLM chat failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate chat response: {str(e)}")

    async def generate_plan(self, task: str, context: str) -> List[Dict[str, Any]]:
        """Generate a plan for executing a task.
        
        Args:
            task: The task description
            context: Additional context for the task
            
        Returns:
            List of steps to execute the task
        """
        prompt = f"""You are Nova, an intelligent browser automation agent. Generate a plan to accomplish this task:

Task: {task}

Context: {context}

Generate a detailed plan with specific steps. Each step should be either a browser action or a tool action.
For browser actions, include 'type', 'selector', and any other required parameters.
For tool actions, include 'tool' name and 'input' parameters.

IMPORTANT: The plan must be a valid JSON array. Do not include any text before or after the array.
Each step must have a 'type' field that is either 'browser' or 'tool'.
For browser actions, include 'action' with 'type' and other required fields.
For tool actions, include 'tool' and 'input' fields.

Example format:
[
    {{"type": "browser", "action": {{"type": "navigate", "url": "https://example.com"}}}},
    {{"type": "browser", "action": {{"type": "click", "selector": "#submit-button"}}}},
    {{"type": "tool", "tool": "screenshot", "input": {{"path": "result.png"}}}}
]

Generate the plan as a JSON array:"""

        try:
            response = await self.generate(prompt)
            # Extract JSON array from response
            try:
                # Find the first '[' and last ']' to extract the JSON array
                start = response.find('[')
                end = response.rfind(']') + 1
                if start >= 0 and end > start:
                    plan_json = response[start:end]
                    plan = json.loads(plan_json)
                    if isinstance(plan, list):
                        # Validate each step
                        for step in plan:
                            if not isinstance(step, dict):
                                raise ValueError("Each step must be a dictionary")
                            if "type" not in step:
                                raise ValueError("Each step must have a 'type' field")
                            if step["type"] not in ["browser", "tool"]:
                                raise ValueError("Step type must be 'browser' or 'tool'")
                            if step["type"] == "browser" and "action" not in step:
                                raise ValueError("Browser steps must have an 'action' field")
                            if step["type"] == "tool" and ("tool" not in step or "input" not in step):
                                raise ValueError("Tool steps must have 'tool' and 'input' fields")
                        return plan
            except Exception as e:
                logger.error(f"Failed to parse plan JSON: {e}", exc_info=True)
                logger.error(f"Raw response: {response}")
            
            raise RuntimeError("Failed to generate valid plan")
        except Exception as e:
            logger.error(f"Plan generation failed: {e}", exc_info=True)
            raise
            
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results.
        
        Args:
            task: The original task
            plan: The executed plan with results
            context: Additional context
            
        Returns:
            Generated response summarizing the results
        """
        prompt = f"""You are Nova, an intelligent browser automation agent. Generate a response for this completed task:

Task: {task}

Context: {context}

Executed Plan:
{json.dumps(plan, indent=2)}

Generate a natural language response that:
1. Summarizes what was done
2. Explains any errors or issues encountered
3. Provides the final result or outcome

Response:"""

        try:
            response = await self.generate(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            raise 