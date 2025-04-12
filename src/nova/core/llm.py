from __future__ import annotations

"""Language model integration."""

import json
import logging
import re

from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import urlparse, urlunparse

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, validator

from .nim_provider import NIMProvider
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

Available tools:
- navigate: Navigate to a URL (input: url)
- click: Click an element matching a selector (input: selector)
- type: Type text into an element (input: selector, text)
- wait: Wait for an element to appear (input: selector, timeout)
- screenshot: Take a screenshot of the current page (input: path)

Generate a detailed plan with specific steps. Each step should use one of the available tools.
For each step, specify the tool name and its required input parameters.

IMPORTANT: The plan must be a valid JSON array or object with a 'steps' field containing an array.
Each step must include the tool name and its input parameters.

Example formats:
[
    {"tool": "navigate", "input": {"url": "https://example.com"}},
    {"tool": "click", "input": {"selector": "#submit-button"}},
    {"tool": "screenshot", "input": {"path": "result.png"}}
]

OR

{
    "steps": [
        {"tool": "navigate", "input": {"url": "https://example.com"}},
        {"tool": "click", "input": {"selector": "#submit-button"}},
        {"tool": "screenshot", "input": {"path": "result.png"}}
    ]
}

Generate the plan:"""

        try:
            response = await self.model.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON from response
            try:
                # Find JSON content (array or object)
                json_match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON content found in response")
                
                plan_json = json_match.group(0)
                plan_data = json.loads(plan_json)
                
                # Handle both array and object formats
                steps = plan_data if isinstance(plan_data, list) else plan_data.get("steps", [])
                if not isinstance(steps, list):
                    raise ValueError("Plan must be a list of steps or an object with a 'steps' array")
                
                # Normalize and validate each step
                normalized_steps = []
                for step in steps:
                    if not isinstance(step, dict):
                        continue
                        
                    # Extract tool and input information
                    tool_name = step.get("tool")
                    if not tool_name:
                        # Try to extract from action if present
                        action = step.get("action", {})
                        if isinstance(action, dict):
                            tool_name = action.get("type")
                    
                    # Get input parameters
                    input_params = step.get("input", {})
                    if not input_params and isinstance(step.get("action"), dict):
                        input_params = {k: v for k, v in step["action"].items() if k != "type"}
                    
                    if tool_name and isinstance(input_params, dict):
                        normalized_step = {
                            "type": "tool",
                            "tool": tool_name,
                            "input": input_params
                        }
                        normalized_steps.append(normalized_step)
                
                if not normalized_steps:
                    raise ValueError("No valid steps found in plan")
                
                return normalized_steps
                
            except Exception as e:
                logger.error(f"Failed to parse plan JSON: {e}", exc_info=True)
                logger.error(f"Raw response: {response_text}")
                raise RuntimeError(f"Failed to generate valid plan: {str(e)}")
            
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
            response = await self.model.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            raise


class StepParameters(BaseModel):
    """Model for step parameters."""
    url: Optional[str] = None
    selector: Optional[str] = None
    text: Optional[str] = None

    @validator('url')
    def validate_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return v


class Step(BaseModel):
    """Model for a single step in the plan."""
    action: str = Field(..., description="The action to perform (navigate, click, or type)")
    parameters: StepParameters = Field(..., description="Parameters for the action")
    description: str = Field(..., description="Description of what this step does")

    @validator('action')
    def validate_action(cls, v):
        valid_actions = {'navigate', 'click', 'type'}
        if v not in valid_actions:
            raise ValueError(f"Action must be one of {valid_actions}")
        return v

    @validator('parameters')
    def validate_parameters(cls, v, values):
        action = values.get('action')
        if action == 'navigate' and not v.url:
            raise ValueError("URL is required for navigate action")
        elif action == 'click' and not v.selector:
            raise ValueError("Selector is required for click action")
        elif action == 'type' and (not v.selector or not v.text):
            raise ValueError("Selector and text are required for type action")
        return v


class Plan(BaseModel):
    """Model for the complete plan."""
    steps: List[Step] = Field(..., description="List of steps to execute")
    reasoning: str = Field(..., description="Explanation of the plan")


class PlanResponse(BaseModel):
    """Model for the complete plan response."""
    status: str = Field(..., description="Status of the plan generation")
    plan: Plan = Field(..., description="The generated plan")
    error: Optional[str] = Field(None, description="Error message if status is not success")

    @validator('status')
    def validate_status(cls, v):
        if v not in {'success', 'error'}:
            raise ValueError("Status must be either 'success' or 'error'")
        return v


def format_url(url: str) -> str:
    """Format URL to ensure consistency."""
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            parsed = urlparse(f"https://{url}")
        return urlunparse(parsed)
    except Exception as e:
        logger.error(f"Error formatting URL {url}: {str(e)}")
        return url


def extract_json_from_response(response_text: str, max_retries: int = 3) -> Optional[Dict]:
    """Extract and validate JSON from LLM response with retries."""
    for attempt in range(max_retries):
        try:
            # Try to find JSON in the response
            json_match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
            if not json_match:
                logger.warning(f"No JSON found in response (attempt {attempt + 1}/{max_retries})")
                continue

            json_str = json_match.group(1)
            # Try to parse the JSON
            try:
                data = json.loads(json_str)
                return data
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in response (attempt {attempt + 1}/{max_retries}): {str(e)}")
                # Try to fix common JSON issues
                json_str = json_str.replace("'", '"')  # Replace single quotes with double quotes
                try:
                    data = json.loads(json_str)
                    return data
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.error(f"Error extracting JSON (attempt {attempt + 1}/{max_retries}): {str(e)}")
            continue

    return None


def validate_plan_data(plan_data: Dict) -> Optional[PlanResponse]:
    """Validate and convert plan data to PlanResponse model."""
    try:
        return PlanResponse(**plan_data)
    except Exception as e:
        logger.error(f"Error validating plan data: {str(e)}")
        return None


class LLM:
    """Language model interface for Nova."""
    
    def __init__(
        self,
        provider: str = "nim",  # Default to NIM provider
        docker_image: str = "nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1:latest",
        api_base: str = "http://localhost:8000",
        model_name: str = "nvidia/llama-3.3-nemotron-super-49b-v1",
        batch_size: int = 4,
        enable_streaming: bool = True,
        **kwargs: Any
    ) -> None:
        """Initialize the language model.
        
        Args:
            provider: LLM provider to use ("nim" or "ollama")
            docker_image: Docker image for NIM service
            api_base: Base URL for NIM API
            model_name: Name of the model to use
            batch_size: Maximum number of requests to process in parallel
            enable_streaming: Whether to enable response streaming
            **kwargs: Additional provider-specific arguments
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self._batch_size = batch_size
        self._enable_streaming = enable_streaming
        self._last_response = None  # Initialize last response
        
        # Initialize the appropriate provider
        if self.provider == "nim":
            self._provider = NIMProvider(
                docker_image=docker_image,
                api_base=api_base,
                model_name=model_name,
                batch_size=batch_size,
                enable_streaming=enable_streaming,
                **kwargs
            )
        elif self.provider == "ollama":
            self._provider = LlamaModel(
                model_name=model_name,
                batch_size=batch_size,
                enable_streaming=enable_streaming,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate a response from the model."""
        response = await self._provider.generate(prompt, max_tokens=max_tokens)
        self._last_response = response
        return response
    
    async def generate_plan(self, task: str, context: str) -> List[Dict[str, Any]]:
        """Generate a plan for executing a task."""
        plan = await self._provider.generate_plan(task, context)
        self._last_response = plan
        return plan
    
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results."""
        response = await self._provider.generate_response(task, plan, context)
        self._last_response = response
        return response
    
    async def generate_batch(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """Generate responses for multiple prompts in parallel."""
        return await self._provider.generate_batch(prompts, **kwargs)
    
    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Generate a streaming response from the model."""
        return await self._provider.generate_stream(prompt, **kwargs)
    
    def get_token_count(self) -> int:
        """Get the current token count."""
        return self._provider.get_token_count()
    
    def reset_token_count(self) -> None:
        """Reset the token count."""
        self._provider.reset_token_count()
    
    def get_last_response(self) -> Optional[Any]:
        """Get the last response from the model.
        
        Returns:
            The last response generated by any of the generate methods
        """
        return self._last_response 