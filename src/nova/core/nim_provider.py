"""NVIDIA NIM API integration."""

import json
import logging
import os
from typing import Any, Dict, List, Optional, AsyncIterator
from datetime import datetime, timedelta
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field
import re

from .language_model import LanguageModel
from .monitoring import PerformanceMonitor

# Configure logging
logger = logging.getLogger(__name__)

# Create a global performance monitor instance
monitor = PerformanceMonitor()

class NIMConfig(BaseModel):
    """Configuration for NVIDIA NIM model."""
    docker_image: str = Field(
        default="nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1:latest",
        description="Docker image for NIM service"
    )
    api_base: str = Field(
        default="https://api.nvcf.nvidia.com/v2/nvcf",
        description="Base URL for NIM API"
    )
    model_name: str = Field(
        default="nvidia/llama-3.3-nemotron-super-49b-v1",
        description="Name of the model to use"
    )
    temperature: float = Field(
        default=0.2,
        description="Temperature for sampling"
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum number of tokens to generate"
    )
    top_p: float = Field(
        default=0.9,
        description="Top-p sampling parameter"
    )
    top_k: int = Field(
        default=50,
        description="Top-k sampling parameter"
    )
    repetition_penalty: float = Field(
        default=1.1,
        description="Penalty for repeating tokens"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed requests"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Initial delay between retries in seconds"
    )

class NIMProvider(LanguageModel):
    """NVIDIA NIM API implementation."""

    def __init__(
        self,
        docker_image: str = "nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1:latest",
        api_base: str = "https://api.nvcf.nvidia.com/v2/nvcf",
        model_name: str = "llama-3.3-nemotron-super-49b-v1",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        batch_size: int = 4,
        enable_streaming: bool = True,
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

    async def generate_plan(self, task: str, context: str) -> tuple[str, List[Dict[str, Any]]]:
        """Generate a plan (thought and action) for executing a task using the NIM API."""
        task_str = str(task) if task is not None else ""
        context_str = str(context) if context is not None else ""
        
        # --- Use the ReAct Prompt (copied from LangChainAdapter, ensure correct formatting) ---
        # Use triple quotes for the main f-string
        prompt = f"""You are Nova, an intelligent browser automation agent designed to accomplish tasks by interacting with web pages using available tools.

Your goal is to generate a plan to accomplish the given task based on the current context, **including the structure of the current web page**. Follow the Thought-Action format:

1.  **Thought:** Briefly explain your reasoning process. 
    - Analyze the `Initial Task Context` and `Recent Execution History`.
    - **Critically examine the `Current Page Structure (After Previous Action)` JSON.** 
    - Note the outcome (`observation`) of the last step in the history, including any errors or the path to a `screenshot` taken after the action.
    - Determine the single best next step towards the `Task` goal based *primarily* on the current page structure and history.
    - If the goal is achieved according to the history and current page state, explain why and use the 'finish' tool.
2.  **Action:** Generate the plan as a JSON array containing the **single next step** (a tool call or the 'finish' action).

Task: {task_str}

Context:
```
{context_str}
```
*Note: The context above includes initial context, the current page structure **captured after the previous action completed**, and recent history. Each history entry includes 'thought', 'action', and 'observation'. The observation contains the action's 'status', 'result' or 'error', and potentially a 'screenshot' file path.* 

Available tools:
- navigate: Navigate to a URL (input: {{'url': '...'}})
- click: Click an element matching a selector (input: {{'selector': '...'}})
- type: Type text into an element (input: {{'selector': '...', 'text': '...'}})
- wait: Wait for an element to appear (input: {{'selector': '...', 'timeout': ...}})
- screenshot: Take a screenshot (input: {{'path': '...'}})
- finish: Use this tool **only** when the task goal is fully achieved. (input: {{'reason': '...'}} - Optional reason for completion)

Output Format:
Your final output **must** be a valid JSON object containing two keys: 'thought' and 'plan'.
The 'thought' value should be a string containing your reasoning.
The 'plan' value should be a JSON array containing **exactly one** step object (either a tool action or the 'finish' action).

Example (Action Step):
```json
{{
  "thought": "The task is to click the login button. Looking at the `Current Page Structure (After Previous Action)`, I see a button with `text: 'Login'` and `attributes: {{'id': 'login-btn'}}`. I will use the click tool with the selector '#login-btn'.",
  "plan": [
    {{"tool": "click", "input": {{"selector": "#login-btn"}}}}
  ]
}}
```

Example (Finish Step):
```json
{{
  "thought": "The user asked for the page title, and the context shows the title is 'Example Domain'. The task is complete.",
  "plan": [
    {{"tool": "finish", "input": {{"reason": "Found the page title as requested."}}}}
  ]
}}
```

Generate the thought and the single next step (or finish action) for the task:"""
        # --- End Prompt ---

        try:
            # Use the standard generate method which now handles nvext correctly
            response_text = await self.generate(prompt)
            if not response_text:
                logger.warning("Invalid response (empty) from NIM API, returning empty plan")
                return "", []
            
            # Parse the response expecting the {"thought": ..., "plan": [...]} structure
            thought, plan_steps = self._parse_thought_plan_response(response_text)
            logger.info("Successfully parsed thought and plan steps.")
            return thought, plan_steps
            
        except Exception as e:
            # Log the specific error during generation or parsing
            logger.error(f"Failed to generate or parse plan with NIM: {e}", exc_info=True)
            # Return empty thought and plan on any failure during this process
            return "", []

    # Remove the old _get_default_plan method as fallback now returns empty
    # def _get_default_plan(self, task: str) -> List[Dict[str, Any]]: ... 

    # Remove the old _parse_plan_response method
    # def _parse_plan_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]: ...

    # --- Add new parsing method for Thought/Plan JSON ---
    def _parse_thought_plan_response(self, response_text: str) -> tuple[str, List[Dict[str, Any]]]:
        """Parse the LLM response expecting {'thought': ..., 'plan': [...]} JSON.
           Extracts the JSON object, trying ```json first, then generic {}.
        """
        logger.debug(f"Attempting to parse thought/plan response (len={len(response_text)} chars)")
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
            
            thought = parsed_data.get("thought", "") 
            plan_steps = parsed_data.get("plan", []) 

            if not isinstance(plan_steps, list):
                raise ValueError("The 'plan' field must contain a list of steps")
            
            # Basic validation of steps (can be enhanced later)
            valid_steps = []
            for step in plan_steps:
                if isinstance(step, dict) and "tool" in step:
                    valid_steps.append({
                        "tool": step.get("tool"),
                        "input": step.get("input", {})
                    })
                else:
                    logger.warning(f"Skipping invalid step format in plan: {step}")
            
            # --- Fix Misleading Log --- 
            logger.info(f"Successfully parsed thought and {len(valid_steps)} plan steps.") # Moved inside try
            return thought, valid_steps

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse thought/plan JSON: {e}", exc_info=True)
            # Log the specific JSON string that failed parsing, if found
            if plan_json:
                 logger.error(f"JSON string that failed parsing: {plan_json}")
            else:
                 logger.error(f"Full response text was: {response_text}")
            return "", [] 
    # --- End new parsing method ---

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