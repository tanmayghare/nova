from __future__ import annotations

"""Language model integration."""

from typing import Any, Dict, List, Optional, Union, Protocol, cast

import logging
import json
import asyncio
from datetime import datetime
from langchain_core.language_models.chat_models import BaseChatModel
from ollama import AsyncClient
import re
from pydantic import BaseModel, Field, validator
from urllib.parse import urlparse, urlunparse

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
        model_name: str = "llama3.2:3b-instruct-q8_0",
        batch_size: int = 4,
        enable_streaming: bool = True,
    ) -> None:
        """Initialize the language model.
        
        Args:
            model_name: Name of the model to use
            batch_size: Maximum number of requests to process in parallel
            enable_streaming: Whether to enable response streaming
        """
        self.model_name = model_name
        self.client = AsyncClient()
        self._batch_size = batch_size
        self._enable_streaming = enable_streaming
        self._token_count = 0
        self._last_token_reset = datetime.now()
        
    async def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate a response from the model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
            
        Raises:
            RuntimeError: If model fails to generate response
        """
        try:
            response = await self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                }
            )
            # Ollama response has 'response' key instead of 'content'
            return response["response"]
        except Exception as e:
            logger.error(f"LLM generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate response: {str(e)}")
            
    async def generate_plan(self, task: str, context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Generate a plan for the given task with improved error handling and validation."""
        try:
            # Enhanced prompt with clear format requirements and examples
            prompt = f"""You are a task planning assistant. Generate a plan for the following task:
{task}

You must respond with ONLY a JSON object in the following format, with no additional text or formatting:
{{
    "status": "success",
    "plan": {{
        "steps": [
            {{
                "type": "tool",
                "tool": "navigate",
                "input": {{
                    "url": "https://example.com"
                }}
            }},
            {{
                "type": "tool",
                "tool": "type",
                "input": {{
                    "selector": ".search-input",
                    "text": "search query"
                }}
            }},
            {{
                "type": "tool",
                "tool": "click",
                "input": {{
                    "selector": ".submit-button"
                }}
            }}
        ],
        "reasoning": "Explanation of the plan steps"
    }}
}}

Rules:
1. Respond with ONLY the JSON object, no other text
2. Each step must have exactly these fields: type, tool, input
3. The type field must be "tool"
4. Valid tools are: navigate, click, type
5. For navigate tool: input must have url
6. For click tool: input must have selector
7. For type tool: input must have both selector and text
8. All URLs must include https:// or http://
9. All selectors must be valid CSS selectors"""

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response_text = await self.generate(prompt)
                    logger.info(f"Raw LLM response (attempt {attempt + 1}): {response_text}")

                    # Extract JSON from the response
                    try:
                        # Try to find JSON in the response
                        json_match = re.search(r'(\{[\s\S]*\})', response_text)
                        if json_match:
                            json_str = json_match.group(1)
                            plan_data = json.loads(json_str)
                        else:
                            logger.warning(f"No JSON found in response (attempt {attempt + 1})")
                            continue
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON (attempt {attempt + 1}): {str(e)}")
                        continue

                    # Validate the plan structure
                    if not isinstance(plan_data, dict):
                        logger.warning(f"Plan is not a dictionary (attempt {attempt + 1})")
                        continue

                    if "status" not in plan_data or plan_data["status"] != "success":
                        logger.warning(f"Invalid status (attempt {attempt + 1})")
                        continue

                    if "plan" not in plan_data:
                        logger.warning(f"Missing plan field (attempt {attempt + 1})")
                        continue

                    plan = plan_data["plan"]
                    if not isinstance(plan, dict) or "steps" not in plan:
                        logger.warning(f"Invalid plan format: missing steps")
                        continue

                    steps = plan["steps"]
                    if not isinstance(steps, list):
                        logger.warning(f"Invalid steps format (attempt {attempt + 1})")
                        continue

                    # Validate each step
                    valid_steps = True
                    for i, step in enumerate(steps):
                        if not isinstance(step, dict):
                            logger.warning(f"Invalid step format at index {i} (attempt {attempt + 1})")
                            valid_steps = False
                            break

                        if "type" not in step or step["type"] != "tool":
                            logger.warning(f"Missing or invalid type in step {i} (attempt {attempt + 1})")
                            valid_steps = False
                            break

                        if "tool" not in step or not isinstance(step["tool"], str):
                            logger.warning(f"Missing or invalid tool in step {i} (attempt {attempt + 1})")
                            valid_steps = False
                            break

                        if "input" not in step or not isinstance(step["input"], dict):
                            logger.warning(f"Missing or invalid input in step {i} (attempt {attempt + 1})")
                            valid_steps = False
                            break

                        # Validate tool-specific parameters
                        tool = step["tool"].lower()
                        input_params = step["input"]
                        
                        if tool == "navigate":
                            if "url" not in input_params:
                                logger.warning(f"Missing url parameter for navigate tool in step {i}")
                                valid_steps = False
                                break
                            # Ensure URL has protocol
                            url = input_params["url"]
                            if not url.startswith(("http://", "https://")):
                                input_params["url"] = f"https://{url}"
                            
                        elif tool == "click":
                            if "selector" not in input_params:
                                logger.warning(f"Missing selector parameter for click tool in step {i}")
                                valid_steps = False
                                break
                            
                        elif tool == "type":
                            if "selector" not in input_params or "text" not in input_params:
                                logger.warning(f"Missing selector or text parameter for type tool in step {i}")
                                valid_steps = False
                                break
                        else:
                            logger.warning(f"Invalid tool type '{tool}' in step {i}")
                            valid_steps = False
                            break

                    if not valid_steps:
                        continue

                    # Return just the steps list to match the protocol
                    return steps

                except Exception as e:
                    logger.error(f"Error generating plan (attempt {attempt + 1}): {str(e)}")
                    if attempt == max_retries - 1:
                        raise

            raise Exception("Failed to generate valid plan after multiple attempts")

        except Exception as e:
            logger.error(f"Error in generate_plan: {str(e)}")
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
            return await self.generate(prompt)
        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            raise 