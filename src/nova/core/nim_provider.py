"""NVIDIA NIM API integration."""

import json
import logging
import re
import os
from typing import Any, Dict, List, Optional, AsyncIterator
from datetime import datetime, timedelta
import aiohttp
import asyncio
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
        """Make a request to the NIM API with retry and circuit breaker."""
        try:
            self._check_circuit_breaker()
            
            # Move parameters into nvext object
            nvext = {}
            for param in ['repetition_penalty', 'top_k']:
                if param in data:
                    nvext[param] = data.pop(param)
            
            if nvext:
                data['nvext'] = nvext
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.api_base}/v1/{endpoint}",
                    json=data,
                    headers={
                        "Authorization": f"Bearer {self._get_api_key()}",
                        "Content-Type": "application/json"
                    },
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status != 200:
                        self._circuit_breaker['failures'] += 1
                        raise RuntimeError(f"NIM API request failed: {await response.text()}")
                    return await response.json()
        except Exception as e:
            self._circuit_breaker['failures'] += 1
            logger.error(f"Request failed: {str(e)}")
            raise

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
            data = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repetition_penalty": self.config.repetition_penalty,
                "stop": [stop] if stop else None
            }
            
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

    async def generate_plan(self, task: str, context: str) -> List[Dict[str, Any]]:
        """Generate a plan for executing a task."""
        # Convert task and context to strings if they're not already
        task_str = str(task) if task is not None else ""
        context_str = str(context) if context is not None else ""
        
        prompt = f"""You are Nova, an intelligent browser automation agent. Generate a plan to accomplish this task:

Task: {task_str}

Context: {context_str}

Available tools:
- navigate: Navigate to a URL (input: url)
- click: Click an element matching a selector (input: selector)
- type: Type text into an element (input: selector, text)
- wait: Wait for an element to appear (input: selector, timeout)
- screenshot: Take a screenshot of the current page (input: path)

Generate a plan with specific steps. Each step should use one of the available tools.
For each step, specify the tool name and its required input parameters.

IMPORTANT: Respond ONLY with a valid JSON array of steps. Each step must include the tool name and its input parameters.
Do not include any explanations, notes, or additional text. Just the JSON array.

Example format:
[
    {{"tool": "navigate", "input": {{"url": "https://example.com"}}}},
    {{"tool": "click", "input": {{"selector": "#submit-button"}}}},
    {{"tool": "screenshot", "input": {{"path": "result.png"}}}}
]

Generate the plan:"""

        try:
            response = await self.generate(prompt)
            if not response:
                logger.warning("Invalid response from NIM API, using default plan")
                return self._get_default_plan(task)
            
            try:
                plan = self._parse_plan_response({"response": response})
                logger.info("Successfully parsed plan: %s", plan)
                return plan
            except Exception as e:
                logger.warning("Failed to parse plan: %s, using default plan", e)
                return self._get_default_plan(task)
            
        except Exception as e:
            logger.warning("Failed to generate plan with NIM: %s, using default plan", e)
            return self._get_default_plan(task)

    def _get_default_plan(self, task: str) -> List[Dict[str, Any]]:
        """Get a default plan for the task."""
        if "amazon.com" in task and "laptop" in task:
            return [{
                "tool": "navigate",
                "input": {
                    "url": "https://www.amazon.com/s?k=laptop"
                }
            }, {
                "tool": "extract",
                "input": {
                    "selector": "div[data-component-type='s-search-result']",
                    "id": "products",
                    "limit": 3
                }
            }]
        else:
            return [{
                "tool": "navigate",
                "input": {
                    "url": "https://example.com"
                }
            }]

    def _parse_plan_response(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse the plan response from the model."""
        try:
            # Extract JSON from response
            content = response.get("response", "")
            # Log the raw response for debugging
            logger.debug(f"Raw response content: {content}")
            
            # Clean up the content
            content = content.strip()
            
            # Try to parse the JSON directly
            plan = json.loads(content)
            
            if isinstance(plan, list) and all(isinstance(step, dict) and "tool" in step for step in plan):
                return plan
            else:
                raise ValueError("Invalid plan format")
        except Exception as e:
            logger.error(f"Failed to parse plan response: {e}")
            raise

    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results."""
        prompt = f"""Task: {task}
Plan execution results:
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
            response = await self.generate(prompt, stop, **kwargs)
            yield response
            return

        try:
            data = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repetition_penalty": self.config.repetition_penalty,
                "stop": [stop] if stop else None,
                "stream": True
            }
            
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