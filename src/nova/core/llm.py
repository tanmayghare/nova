from __future__ import annotations

"""Language model integration."""

import json
import logging
import re
import asyncio

from typing import Any, Dict, List, Optional, Protocol
from urllib.parse import urlparse, urlunparse

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, validator

from .nim_provider import NIMProvider
from .llama import LlamaModel

logger = logging.getLogger(__name__)


class LanguageModel(Protocol):
    """Protocol for language models."""
    
    async def generate_plan(self, task: str, context: str) -> tuple[str, List[Dict[str, Any]]]:
        """Generate a plan (including thought) for executing a task."""
        ...
    
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results."""
        ...


class LangChainAdapter:
    """Adapter for LangChain models to implement the LanguageModel protocol."""
    
    def __init__(self, model: BaseChatModel):
        """Initialize with a LangChain model."""
        self.model = model
    
    async def generate_plan(self, task: str, context: str) -> tuple[str, float, List[Dict[str, Any]]]:
        """Generate a plan for executing a task, including the thought process and confidence score.
        
        Args:
            task: The task description
            context: Additional context for the task
            
        Returns:
            A tuple containing the thought (str), confidence (float), and the list of steps (List[Dict]).
        """
        prompt = f"""You are Nova, an intelligent browser automation agent designed to accomplish tasks by interacting with web pages using available tools.

Your goal is to generate a plan to accomplish the given task based on the current context, **including the structure of the current web page**. Follow the Thought-Action format, **including a confidence score for your proposed action**:

1.  **Thought:** Briefly explain your reasoning process. 
    - Analyze the `Initial Task Context` and `Recent Execution History`.
    - **Critically examine the `Current Page Structure (After Previous Action)` JSON.** 
    - Note the outcome (`observation`) of the last step in the history. 
        - **If the last observation status was 'error':** Analyze the `error` message. Based on the error and the current page structure, decide on a corrective action (e.g., try a different selector, wait, try a different tool) or conclude with `finish` if recovery is impossible.
        - **If the last observation status was 'low_confidence_retry':** Re-evaluate the goal using the potentially added 'Extended Context (Full HTML)' and propose a higher-confidence action or `finish`.
        - **Otherwise (success or first step):** Determine the single best next step towards the `Task` goal based on the current page structure and history.
    - If the overall task goal is achieved according to the history and current page state, explain why and use the 'finish' tool.
2.  **Confidence Score:** Provide a numerical score (0.0 to 1.0) indicating your confidence that the proposed action is correct and will succeed towards the goal. 
    - Base confidence on: clarity of the goal, uniqueness/reliability of selectors (if applicable), consistency with history, likelihood of achieving the task objective with this step.
    - Use lower scores if selectors are ambiguous, the action seems risky, or the goal is unclear.
3.  **Action:** Generate the plan as a JSON array containing the **single next step** (a tool call or the 'finish' action).

Task: {task}

Context:
```
{context}
```
*Note: The context above includes initial context, the current page structure captured after the previous action completed, and recent history. Each history entry includes 'thought', 'action', 'confidence' (if available), and 'observation'. The observation contains the action's 'status' ('success', 'error', 'low_confidence_retry', 'halted'), 'result' or 'error', and potentially a 'screenshot' file path. If the previous step had low confidence, the context might also include an 'Extended Context (Full HTML from Previous Step)' section.* 

Available tools:
- navigate: Navigate to a URL (input: {{'url': '...'}})
- click: Click an element matching a selector (input: {{'selector': '...'}})
- type: Type text into an element (input: {{'selector': '...', 'text': '...'}})
- wait: Wait for an element to appear (input: {{'selector': '...', 'timeout': ...}})
- screenshot: Take a screenshot (input: {{'path': '...'}})
- finish: Use this tool **only** when the task goal is fully achieved. (input: {{'reason': '...'}} - Optional reason for completion)
# Add other tools dynamically if needed

Output Format:
Your final output **must** be a valid JSON object containing three keys: 'thought' (string), 'confidence' (float between 0.0 and 1.0), and 'plan' (JSON array with one step).

Example (Action Step):
```json
{{
  "thought": "The task is to click the login button. Looking at the `Current Page Structure (After Previous Action)`, I see a button with `text: 'Login'` and a unique `id: 'login-btn'`. This seems unambiguous.",
  "confidence": 0.95,
  "plan": [
    {{\"tool\": \"click\", \"input\": {{\"selector\": \"#login-btn\"}}}}
  ]
}}
```

Example (Corrective Step after Error):
```json
{{
  "thought": "The previous action `click` with selector '#submit' failed with error 'Timeout waiting for element'. The current page structure still shows the button exists. I will try waiting for the element to be clickable first, then click again.",
  "confidence": 0.8,
  "plan": [
    {{\"tool\": \"wait\", \"input\": {{"selector": \"#submit\", \"timeout\": 15}}}}
    // Note: Ideally the LLM would then plan the click in the *next* step after the wait succeeds.
    // For now, we expect one action per step.
  ]
}}
```

Example (Finish Step):
```json
{{
  "thought": "The user asked for the page title, and the context shows the title is 'Example Domain'. The task is complete.",
  "confidence": 1.0,
  "plan": [
    {{\"tool\": \"finish\", \"input\": {{\"reason\": \"Found the page title as requested.\"}}}}
  ]
}}
```

Generate the thought, confidence score, and the single next step (or finish action) for the task:"""

        try:
            response = await self.model.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract JSON object from response
            plan_json = None
            try:
                # Priority 1: Look for ```json ... ``` block
                json_block_match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
                if json_block_match:
                    plan_json = json_block_match.group(1).strip()
                    logger.debug("Extracted JSON using ```json block.")
                else:
                    # Priority 2: Look for the first generic { ... } block
                    generic_match = re.search(r"\{\s*.*?\s*\}", response_text, re.DOTALL)
                    if generic_match:
                        plan_json = generic_match.group(0).strip()
                        logger.debug("Extracted JSON using generic {.*?} block.")
                    else:
                         raise ValueError("No JSON object found in response text using ```json or {.*?} patterns.")

                logger.debug(f"Extracted JSON string: {plan_json[:500]}...")
                parsed_data = json.loads(plan_json)
                
                if not isinstance(parsed_data, dict):
                    raise ValueError("Parsed JSON is not a dictionary")

                # Extract thought, confidence, and plan
                thought = parsed_data.get("thought", "") 
                plan_steps = parsed_data.get("plan", []) 
                confidence = 0.0 # Default confidence
                raw_confidence = parsed_data.get("confidence")
                if isinstance(raw_confidence, (float, int)):
                    confidence = max(0.0, min(1.0, float(raw_confidence)))
                elif isinstance(raw_confidence, str):
                    try:
                        confidence = max(0.0, min(1.0, float(raw_confidence)))
                    except ValueError:
                        logger.warning(f"Could not parse confidence score string: '{raw_confidence}'. Defaulting to 0.0.")
                elif raw_confidence is not None:
                     logger.warning(f"Unexpected type for confidence score: {type(raw_confidence)}. Defaulting to 0.0.")

                if not isinstance(plan_steps, list):
                    raise ValueError("The 'plan' field must contain a list of steps")
                
                # Normalize and validate steps (no major change needed here for finish tool)
                normalized_steps = []
                for step in plan_steps:
                    if not isinstance(step, dict):
                        logger.warning(f"Skipping invalid step format: {step}")
                        continue
                        
                    tool_name = step.get("tool")
                    input_params = step.get("input", {})

                    if not tool_name:
                         logger.warning(f"Skipping step missing 'tool' key: {step}")
                         continue 
                    
                    if not isinstance(input_params, dict):
                         # Allow non-dict input for finish if needed, but generally expect dict
                         if tool_name != "finish":
                             logger.warning(f"Input for tool '{tool_name}' is not a dictionary: {input_params}. Using empty input.")
                             input_params = {} 
                         elif input_params is None: # Allow null input for finish
                              input_params = {}
                         elif not isinstance(input_params, dict): # Treat other non-dicts as error for finish? Or empty?
                              logger.warning(f"Non-dict input for finish tool: {input_params}. Using empty input.")
                              input_params = {}

                    normalized_step = {
                        "tool": tool_name,
                        "input": input_params
                    }
                    normalized_steps.append(normalized_step)
                
                logger.info(f"Successfully parsed thought, confidence={confidence:.2f}, and {len(normalized_steps)} plan steps.")
                # Return the 3-tuple
                return thought, confidence, normalized_steps 
                
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.error(f"Failed to parse LLM response JSON: {e}", exc_info=True)
                if plan_json:
                     logger.error(f"JSON string that failed parsing: {plan_json}")
                else:
                     logger.error(f"Raw response potentially causing error: {response_text}")
                # Return empty thought, 0.0 confidence, empty plan on failure
                return "", 0.0, [] 

        except Exception as e:
            logger.error(f"Plan generation failed: {e}", exc_info=True)
            # Return empty on LLM communication failure
            return "", 0.0, [] 
    
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results (ReAct history).
        
        Args:
            task: The original task description
            plan: The full ReAct action history (list of thought/action/observation dicts)
            context: The final context string used before generating this response (optional context)
            
        Returns:
            Generated response summarizing the execution
        """
        # Use the action_history (passed as 'plan') as the primary source
        history_json = json.dumps(plan, indent=2) 
        
        prompt = f"""You are Nova, an intelligent browser automation agent. The following task was attempted:

Task: {task}

The execution involved the following sequence of thoughts, actions, and observations (ReAct History):
```json
{history_json}
```

Generate a concise natural language response for the user that:
1. Summarizes the key actions taken based on the history.
2. Mentions any significant errors encountered during execution (from observations).
3. States the final outcome or result (if discernible from the history or cumulative results).

Response:"""

        try:
            response = await self.model.ainvoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Response generation failed: {e}", exc_info=True)
            raise # Re-raise for now, agent might handle it


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
    
    async def generate_plan(self, task: str, context: str) -> tuple[str, float, List[Dict[str, Any]]]:
        """Generate a plan (including thought and confidence) for executing a task."""
        # Ensure the provider method signature matches if it exists
        # If _provider implements generate_plan directly, it needs updating too.
        # Assuming it uses the LangChainAdapter or NIMProvider logic internally or 
        # the call adapts. If not, the specific provider code needs changing.
        if hasattr(self._provider, 'generate_plan') and asyncio.iscoroutinefunction(self._provider.generate_plan):
             thought, confidence, plan = await self._provider.generate_plan(task, context)
        else:
             # Fallback or error if provider doesn't support this signature
             # This might require adjustment based on specific provider implementations
             # For now, assume LangChainAdapter handles it or similar logic applies
             logger.warning(f"Provider {self.provider} might not directly support returning thought/confidence. Attempting generation.")
             # Simulate the call if direct method is missing (adapt as needed)
             # This part is speculative and depends on the provider's design
             # You might need to instantiate LangChainAdapter here if self._provider is just the raw model
             if isinstance(self._provider, BaseChatModel): # Example check
                 adapter = LangChainAdapter(self._provider)
                 thought, confidence, plan = await adapter.generate_plan(task, context)
             else:
                 # Default to empty if provider cannot generate thought/confidence/plan tuple
                 logger.error(f"Provider {self.provider} does not implement the required generate_plan returning (thought, confidence, plan).")
                 thought, confidence, plan = "", 0.0, []

        self._last_response = {"thought": thought, "confidence": confidence, "plan": plan} # Store all three
        return thought, confidence, plan
    
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results."""
        # This method likely needs updating later to handle the ReAct history better
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