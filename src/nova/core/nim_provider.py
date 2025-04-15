"""NVIDIA NIM API integration."""

import os
import logging
import re
import json
import commentjson
import aiohttp
import asyncio
from typing import Any, Dict, List, Optional, AsyncIterator, Tuple
from datetime import datetime, timedelta

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field

from .language_model import LanguageModel
from .monitoring import PerformanceMonitor

# Configure logging
logger = logging.getLogger(__name__)

# Create a global performance monitor instance
monitor = PerformanceMonitor()

class NIMConfig(BaseModel):
    """Configuration for NVIDIA NIM model."""
    docker_image: str = Field(
        default=os.environ.get("NIM_DOCKER_IMAGE"),
        description="Docker image for NIM service"
    )
    api_base: str = Field(
        default=os.environ.get("NIM_API_BASE_URL"),
        description="Base URL for NIM API"
    )
    model_name: str = Field(
        default=os.environ.get("MODEL_NAME"),
        description="Name of the model to use"
    )
    temperature: float = Field(
        default=os.environ.get("MODEL_TEMPERATURE"),
        description="Temperature for sampling"
    )
    max_tokens: int = Field(
        default=os.environ.get("MODEL_MAX_TOKENS"),
        description="Maximum number of tokens to generate"
    )
    top_p: float = Field(
        default=os.environ.get("MODEL_TOP_P"),
        description="Top-p sampling parameter"
    )
    top_k: int = Field(
        default=os.environ.get("MODEL_TOP_K"),
        description="Top-k sampling parameter"
    )
    repetition_penalty: float = Field(
        default=os.environ.get("MODEL_REPETITION_PENALTY"),
        description="Penalty for repeating tokens"
    )
    max_retries: int = Field(
        default=os.environ.get("MODEL_MAX_RETRIES"),
        description="Maximum number of retries for failed requests"
    )
    retry_delay: float = Field(
        default=os.environ.get("MODEL_RETRY_DELAY"),
        description="Initial delay between retries in seconds"
    )

class NIMProvider(LanguageModel):
    """NVIDIA NIM API implementation."""

    def __init__(
        self,
        docker_image: str = os.environ.get("NIM_DOCKER_IMAGE"),
        api_base: str = os.environ.get("NIM_API_BASE_URL"),
        model_name: str = os.environ.get("MODEL_NAME"),
        temperature: float = os.environ.get("MODEL_TEMPERATURE"),
        max_tokens: int = os.environ.get("MODEL_MAX_TOKENS"),
        top_p: float = os.environ.get("MODEL_TOP_P"),
        top_k: int = os.environ.get("MODEL_TOP_K"),
        repetition_penalty: float = os.environ.get("MODEL_REPETITION_PENALTY"),
        max_retries: int = os.environ.get("MODEL_MAX_RETRIES"),
        retry_delay: float = os.environ.get("MODEL_RETRY_DELAY"),
        batch_size: int = os.environ.get("MODEL_BATCH_SIZE"),
        enable_streaming: bool = os.environ.get("MODEL_ENABLE_STREAMING"),
    ) -> None:
        """Initialize the NIM provider.
        
        Args:
            docker_image: Docker image for NIM service
            api_base: Base URL for NIM API
            model_name: Name of the model to use
            temperature: Temperature for sampling
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            max_retries: Maximum number of retries for failed requests
            retry_delay: Initial delay between retries in seconds
            batch_size: Maximum number of requests to process in parallel
            enable_streaming: Whether to enable response streaming
        """
        self.config = NIMConfig(
            docker_image=docker_image,
            api_base=api_base,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self._batch_size = batch_size
        self._enable_streaming = enable_streaming
        self._token_count = 0
        self._last_token_reset = datetime.now()
        self._circuit_breaker = {
            'failures': 0,
            'last_failure': None,
            'is_open': False
        }

    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker should be opened/closed."""
        if not self._circuit_breaker['is_open']:
            if self._circuit_breaker['failures'] >= 5:
                self._circuit_breaker['is_open'] = True
                self._circuit_breaker['last_failure'] = datetime.now()
                raise RuntimeError("Circuit breaker opened due to multiple failures")
        else:
            # Check if enough time has passed to close the circuit
            if (datetime.now() - self._circuit_breaker['last_failure']) > timedelta(minutes=5):
                self._circuit_breaker['is_open'] = False
                self._circuit_breaker['failures'] = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _make_request(self, endpoint: str, data: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """Make a request to the NIM API with retry logic and circuit breaking."""
        self._check_circuit_breaker()
        api_key = self._get_api_key()
        url = f"{self.config.api_base}/v1/{endpoint}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            logger.debug(f"NIM Request - URL: {url}")
            # logger.debug(f"NIM Request - Headers: {headers}") # Optional: Log headers (beware of key exposure)
            logger.debug(f"NIM Request - Body: {json.dumps(data)}")
            logger.info(f"NIM Request - Sending POST to {url}...")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    logger.info(f"NIM Response - Received Status: {response.status} from {url}")
                    if response.status != 200:
                        self._circuit_breaker['failures'] += 1
                        response_text = await response.text()
                        logger.error(f"NIM Response - Error Body: {response_text}")
                        raise RuntimeError(f"NIM API request failed ({response.status}): {response_text}")
                    
                    logger.debug("NIM Response - Reading body...")
                    response_json = await response.json()
                    logger.debug(f"NIM Response - Body JSON: {response_json}")
                    # Reset circuit breaker on success
                    self._circuit_breaker['failures'] = 0
                    self._circuit_breaker['last_attempt'] = datetime.now()
                    return response_json
        except Exception as e:
            self._circuit_breaker['failures'] += 1
            self._circuit_breaker['last_attempt'] = datetime.now()
            logger.error(f"NIM Request - Failed: {str(e)}", exc_info=True)
            raise # Re-raise the exception after logging

    def _get_api_key(self) -> str:
        """Get the NIM API key from environment variables."""
        api_key = os.getenv("NVIDIA_NIM_API_KEY")
        print(f"DEBUG: Fetched NVIDIA_NIM_API_KEY = {api_key}")
        if not api_key:
            raise ValueError("NVIDIA_NIM_API_KEY environment variable not set")
        return api_key

    async def generate(
        self,
        prompt: str,
        stop: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Generate text using the NIM API."""
        try:
            # Prepare base data, using kwargs to override config defaults if provided
            data = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get('temperature', self.config.temperature),
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "stop": [stop] if stop else kwargs.get('stop'), # Allow stop via arg or kwargs
                # Do not include stream: True here for non-streaming generate
            }

            # Handle nvext parameters
            nvext_params = {}
            top_k_val = kwargs.get('top_k', self.config.top_k)
            if top_k_val is not None: # Check if top_k should be included
                nvext_params["top_k"] = top_k_val
            
            rep_penalty_val = kwargs.get('repetition_penalty', self.config.repetition_penalty)
            if rep_penalty_val is not None: # Check if repetition_penalty should be included
                nvext_params["repetition_penalty"] = rep_penalty_val

            if nvext_params: # Only add nvext if it contains parameters
                data["nvext"] = nvext_params
                
            # Filter out None values from the main data dict AFTER processing
            data = {k: v for k, v in data.items() if v is not None}

            response = await self._make_request("chat/completions", data)
            if "choices" in response and len(response["choices"]) > 0:
                if "message" in response["choices"][0]:
                    return response["choices"][0]["message"]["content"]
                elif "text" in response["choices"][0]:
                    return response["choices"][0]["text"]
            raise ValueError("Invalid response format from NIM API")
                
        except Exception as e:
            logger.error(f"Error generating text: {e}", exc_info=True)
            raise

    async def generate_plan(self, task: str, context: str) -> tuple[str, float, List[Dict[str, Any]]]:
        """Generate a plan (thought and action) for executing a task using the NIM API."""
        task_str = str(task) if task is not None else ""
        context_str = str(context) if context is not None else ""
        
        # --- Use the ReAct Prompt (copied from LangChainAdapter, ensure correct formatting) ---
        # Use triple quotes for the main f-string
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
    **Important: Do not use backticks (`) within the 'thought' string.**
2.  **Confidence Score:** Provide a numerical score (0.0 to 1.0) indicating your confidence that the proposed action is correct and will succeed towards the goal. 
    - Base confidence on: clarity of the goal, uniqueness/reliability of selectors (if applicable), consistency with history, likelihood of achieving the task objective with this step.
    - Use lower scores if selectors are ambiguous, the action seems risky, or the goal is unclear.
3.  **Action:** Generate the plan as a JSON array containing the **single next step** (a tool call or the 'finish' action).

Task: {task_str}

Context:
```
{context_str}
```
*Note: The context above includes initial context, the current page structure captured after the previous action completed, and recent history. Each history entry includes 'thought', 'action', 'confidence' (if available), and 'observation'. The observation contains the action's 'status' ('success', 'error', 'low_confidence_retry', 'halted'), 'result' or 'error', and potentially a 'screenshot' file path. If the previous step had low confidence, the context might also include an 'Extended Context (Full HTML from Previous Step)' section.* 

Available tools:
- navigate: Navigate to a URL (input: {{'url': '...'}})
- click: Click an element matching a selector (input: {{'selector': '...'}})
- type: Type text into an element (input: {{'selector': '...', 'text': '...'}})
- wait: Wait for an element to appear (input: {{'selector': '...', 'timeout': ...}})
- screenshot: Take a screenshot (input: {{'path': '...'}})
- finish: Use this tool **only** when the task goal is fully achieved. (input: {{'reason': '...'}} - Optional reason for completion)

Output Format:
Your final output **must** be a valid JSON object containing three keys: 'thought' (string), 'confidence' (float between 0.0 and 1.0), and 'plan' (JSON array with one step).

Example (Action Step):
```json
{{
  "thought": "The task is to click the login button. Looking at the 'Current Page Structure (After Previous Action)', I see a button with text: 'Login' and a unique id: 'login-btn'. This seems unambiguous.",
  "confidence": 0.95,
  "plan": [
    {{"tool": "click", "input": {{"selector": "#login-btn"}}}}
  ]
}}
```

Example (Corrective Step after Error):
```json
{{
  "thought": "The previous action 'click' with selector '#submit' failed with error 'Timeout waiting for element'. The current page structure still shows the button exists. I will try waiting for the element to be clickable first, then click again.",
  "confidence": 0.8,
  "plan": [
    {{"tool": "wait", "input": {{"selector": "#submit", "timeout": 15}}}}
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
    {{"tool": "finish", "input": {{"reason": "Found the page title as requested."}}}}
  ]
}}
```

Generate the thought, confidence score, and the single next step (or finish action) for the task:"""
        # --- End Prompt ---

        try:
            # Use the standard generate method which now handles nvext correctly
            response_text = await self.generate(prompt)
            if not response_text:
                logger.warning("Invalid response (empty) from NIM API, returning empty plan/confidence")
                return "", 0.0, []
            
            # Parsing method now returns the 3-tuple
            thought, confidence, plan_steps = self._parse_thought_plan_response(response_text)
            # Log was moved inside the parser
            return thought, confidence, plan_steps
            
        except Exception as e:
            logger.error(f"Failed to generate or parse plan with NIM: {e}", exc_info=True)
            return "", 0.0, []

    # Remove the old _get_default_plan method as fallback now returns empty
    # def _get_default_plan(self, task: str) -> List[Dict[str, Any]]: ... 

    # Remove the old _parse_plan_response method
    # def _parse_plan_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]: ...

    # --- Add new parsing method for Thought/Plan JSON ---
    def _parse_thought_plan_response(self, response: str) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Parse the LLM's response into thought, confidence, and plan steps."""
        json_str = None
        cleaned_json_str = None
        logger.debug(f"Attempting to parse raw response (len={len(response)}): {response[:500]}...")
        try:
            # Priority 1: Look for ```json ... ``` block
            json_block_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1).strip()
                logger.debug("Extracted JSON using ```json block.")
            else:
                # Priority 2: Look for the first generic { ... } block
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    logger.debug("Extracted JSON using generic {.*?} block.")
                else:
                    raise ValueError("No JSON object delimiters ('{' or '```json') found in response text.")
            
            logger.debug(f"Extracted JSON string: {json_str[:500]}...")

            # Clean up any potential *leading/trailing* whitespace or common LLM artifacts *before* parsing
            # and replace backticks that might confuse commentjson within strings.
            cleaned_json_str = json_str.strip()
            cleaned_json_str = cleaned_json_str.replace('\\`', "'") # Corrected: Replace literal backticks with single quotes
            logger.debug(f"Cleaned JSON string (backticks replaced): {cleaned_json_str[:500]}...")
            
            # Use commentjson.loads
            parsed_data = commentjson.loads(cleaned_json_str)
            
            # --- Validation --- (Keep existing validation)
            if not isinstance(parsed_data, dict):
                 raise ValueError("Parsed JSON is not a dictionary")
            
            # ... (rest of the validation for thought, confidence, plan) ...
            thought = parsed_data.get("thought", "")
            confidence = 0.0
            raw_confidence = parsed_data.get("confidence")
            # ... (confidence parsing logic) ...
            
            plan_steps_raw = parsed_data.get("plan", [])
            if not isinstance(plan_steps_raw, list):
                raise ValueError("Parsed 'plan' field is not a list.")

            plan_steps = []
            allowed_tools = ['navigate', 'click', 'type', 'wait', 'screenshot', 'finish'] # Ensure finish is allowed
            for step in plan_steps_raw:
                 if not isinstance(step, dict) or 'tool' not in step:
                     logger.warning(f"Skipping invalid step format or missing tool: {step}")
                     continue
                     
                 tool = step['tool']
                 if tool not in allowed_tools:
                     logger.warning(f"Skipping step with invalid tool '{tool}': {step}")
                     continue
                     
                 plan_steps.append({
                     'tool': tool,
                     'input': step.get('input', {}) # Allow empty input
                 })
            
            logger.info(f"Successfully parsed: thought='{thought[:50]}...', confidence={confidence:.2f}, steps={len(plan_steps)}")
            return thought, confidence, plan_steps

        # Catch commentjson exceptions and other errors
        except (commentjson.JSONLibraryException, ValueError, TypeError) as e:
            logger.error(f"Failed to parse thought/plan JSON with commentjson: {e}", exc_info=True)
            logger.error(f"Raw response received: {response}")
            if cleaned_json_str is not None:
                 logger.error(f"Attempted JSON string (after cleaning): {cleaned_json_str}")
            elif json_str is not None:
                 logger.error(f"Attempted JSON string (before cleaning): {json_str}")
            # Re-raise a ValueError consistent with previous error handling
            raise ValueError(f"Failed to parse LLM response JSON: {str(e)}")
        except Exception as e:
            # Catch other potential errors during extraction/parsing/validation
            logger.error(f"Unexpected error parsing thought/plan response: {e}", exc_info=True)
            logger.error(f"Raw response received: {response}")
            # Re-raise a ValueError consistent with previous error handling
            raise ValueError(f"Failed to parse LLM response: {str(e)}")
            
    # --- End parsing method ---

    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        # ... (This method likely needs updating later too, but focus on generate_plan now) ...
        prompt = f"""Task: {task}
ReAct History:
{json.dumps(plan, indent=2)}

Context:
{context}

Please provide a summary of the task execution results.
"""
        return await self.generate(prompt)

    async def generate_batch(
        self,
        prompts: List[str],
        stop: Optional[str] = None,
        **kwargs: Any
    ) -> List[str]:
        """Generate responses for multiple prompts in parallel."""
        results = []
        for i in range(0, len(prompts), self._batch_size):
            batch = prompts[i:i + self._batch_size]
            batch_tasks = [self.generate(p, stop, **kwargs) for p in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        return results

    async def generate_stream(
        self,
        prompt: str,
        stop: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the model."""
        if not self._enable_streaming:
            # If streaming disabled, call regular generate and yield result
            response = await self.generate(prompt, stop, **kwargs)
            yield response
            return

        try:
            # Base data for streaming
            data = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get('temperature', self.config.temperature),
                "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
                "top_p": kwargs.get('top_p', self.config.top_p),
                "stop": [stop] if stop else kwargs.get('stop'),
                "stream": True # Crucial for streaming endpoint
            }

            # Handle nvext parameters for streaming
            nvext_params = {}
            top_k_val = kwargs.get('top_k', self.config.top_k)
            if top_k_val is not None:
                nvext_params["top_k"] = top_k_val
            
            rep_penalty_val = kwargs.get('repetition_penalty', self.config.repetition_penalty)
            if rep_penalty_val is not None:
                nvext_params["repetition_penalty"] = rep_penalty_val

            if nvext_params:
                data["nvext"] = nvext_params

            # Filter out None values
            data = {k: v for k, v in data.items() if v is not None}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.api_base}/v1/chat/completions",
                    json=data,
                    headers={
                        "Authorization": f"Bearer {self._get_api_key()}",
                        "Content-Type": "application/json"
                    }
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(f"NIM API request failed: {await response.text()}")
                    
                    async for line in response.content:
                        if line:
                            try:
                                # Handle both SSE and JSON streaming formats
                                if line.startswith(b"data: "):
                                    chunk = json.loads(line[6:])
                                else:
                                    chunk = json.loads(line)
                                    
                                if "choices" in chunk and chunk["choices"]:
                                    content = chunk["choices"][0].get("delta", {}).get("content", "")
                                    if content:
                                        yield content
                                        self._token_count += 1
                            except json.JSONDecodeError:
                                continue
                            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate streaming response: {str(e)}")

    def get_token_count(self) -> int:
        """Get the current token count."""
        if datetime.now() - self._last_token_reset > timedelta(seconds=300):
            self._token_count = 0
            self._last_token_reset = datetime.now()
        return self._token_count

    def reset_token_count(self) -> None:
        """Reset the token count."""
        self._token_count = 0
        self._last_token_reset = datetime.now()

    async def interpret_command(self, command: str) -> Dict[str, Any]:
        """Interpret a natural language command into a structured plan."""
        prompt = f"""Interpret the following command into a structured plan:
Command: {command}

CRITICAL: Your output MUST be ONLY the JSON object, starting with `{{` and ending with `}}`. 
Do NOT include ```json fences, explanations, apologies, or any other text before or after the JSON object.

The JSON object must have the following keys:
- "goal": (String) The overall goal derived from the command.
- "entities": (List of Strings) Key entities extracted from the command.
- "plan": (List of Objects) A sequence of tool actions to execute. Each action object must have:
  - "action": (String) The type of action (MUST be one of: 'navigate', 'click', 'type', 'wait', 'screenshot').
  - "parameters": (Object) Parameters specific to the action (e.g., {{"url": "..."}} for navigate, {{"selector": "...", "text": "..."}} for type).
  - "reasoning": (String) A brief explanation of why this specific action step is needed to achieve the goal.

Example JSON Output:
{{
  "goal": "Find information about Python web frameworks",
  "entities": ["Python", "web frameworks"],
  "plan": [
    {{
      "action": "navigate",
      "parameters": {{"url": "https://www.google.com"}},
      "reasoning": "Need to start at a search engine to find information"
    }},
    {{
      "action": "type",
      "parameters": {{"selector": "textarea[name='q']", "text": "Python web frameworks"}},
      "reasoning": "Enter the search query into the search bar"
    }},
    {{
       "action": "click",
       "parameters": {{"selector": "input[name='btnK']"}},
       "reasoning": "Click the search button to submit the query"
    }}
  ]
}}

Now, interpret the command: '{command}' and provide ONLY the JSON object."""
        
        response_text = "" # Initialize in case generate fails
        json_str = None # Initialize
        try:
            logger.info(f"Generating interpretation for command: '{command}'")
            response_text = await self.generate(prompt) # Use self.generate

            # Extract JSON from response_text (reuse extraction logic)
            json_block_match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1).strip()
            else:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = response_text[start_idx:end_idx].strip()
            
            if not json_str:
                raise ValueError("No JSON object found in LLM interpretation response.")

            logger.debug(f"Extracted interpretation JSON string: {json_str[:500]}...")
            
            # Use commentjson.loads
            parsed_interpretation = commentjson.loads(json_str)

            # --- Validation --- (Keep existing validation)
            if not isinstance(parsed_interpretation, dict):
                raise ValueError("Interpreted plan is not a dictionary.")
            # ... (rest of validation for goal, entities, plan) ...

            logger.info(f"Successfully interpreted command into plan.")
            return parsed_interpretation

        # Catch commentjson exceptions and other errors
        except (commentjson.JSONLibraryException, ValueError, TypeError) as e:
            logger.error(f"Failed to parse interpreted plan JSON with commentjson: {e}")
            logger.error(f"Raw LLM response was: {response_text}") # Use response_text here
            if json_str:
                 logger.error(f"Attempted JSON string: {json_str}")
            raise ValueError(f"Failed to decode/validate interpreted plan JSON: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error generating or parsing interpreted plan: {e}", exc_info=True)
            logger.error(f"Raw LLM response was: {response_text}") # Use response_text here
            raise ValueError(f"Unexpected error interpreting command: {str(e)}") from e 