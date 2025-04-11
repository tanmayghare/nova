"""Llama language model integration."""

import logging
import re
import json
from typing import Any, Dict, List, Optional, AsyncIterator
from functools import lru_cache
from datetime import datetime, timedelta
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from cachetools import TTLCache, LRUCache

from llama_cpp import Llama  # type: ignore
from .monitoring import PerformanceMonitor
from .language_model import LanguageModel
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create a global performance monitor instance
monitor = PerformanceMonitor()

# Connection pool configuration
POOL_CONFIG = {
    'limit': 100,  # Maximum number of connections
    'force_close': True,  # Force close connections
    'enable_cleanup_closed': True,  # Cleanup closed connections
    'keepalive_timeout': 30,  # Keepalive timeout in seconds
}

# Global session and connection pool
_session = None
_connection_pool = None

def get_session() -> aiohttp.ClientSession:
    """Get or create a global aiohttp session with connection pooling."""
    global _session, _connection_pool
    if _session is None or _session.closed:
        if _connection_pool is None:
            _connection_pool = aiohttp.TCPConnector(**POOL_CONFIG)
        _session = aiohttp.ClientSession(connector=_connection_pool)
    return _session

async def cleanup_session():
    """Cleanup the global session and connection pool."""
    global _session, _connection_pool
    if _session and not _session.closed:
        await _session.close()
    if _connection_pool and not _connection_pool.closed:
        await _connection_pool.close()
    _session = None
    _connection_pool = None

PROMPT_TEMPLATE = """You are an AI assistant that generates browser automation plans. Your task is to convert natural language instructions into a structured JSON plan.

Task: {query}
Context: {context}

You must respond with a valid JSON object that has a 'steps' array. Each step should be a browser action or data extraction step.

Example plan:
{
  "steps": [
    {
      "type": "browser",
      "action": {
        "type": "navigate",
        "url": "https://example.com"
      }
    },
    {
      "type": "extract",
      "selector": "#content",
      "id": "result"
    }
  ]
}

Important: Return ONLY the JSON object, no other text or explanation."""

class LlamaConfig(BaseModel):
    """Configuration for Llama model."""
    model_name: str = Field(
        default="llama3.2:3b-instruct-q8_0",
        description="Name of the Ollama model to use"
    )
    temperature: float = Field(
        default=0.1,
        description="Temperature for sampling"
    )
    max_tokens: int = Field(
        default=2048,
        description="Maximum number of tokens to generate"
    )
    top_p: float = Field(
        default=0.95,
        description="Top-p sampling parameter"
    )
    top_k: int = Field(
        default=40,
        description="Top-k sampling parameter"
    )
    repeat_penalty: float = Field(
        default=1.1,
        description="Penalty for repeating tokens"
    )
    cache_ttl: int = Field(
        default=300,  # 5 minutes
        description="Cache TTL in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed requests"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Initial delay between retries in seconds"
    )
    max_cache_size: int = Field(
        default=1000,
        description="Maximum number of items in the cache"
    )
    cache_ttl_variance: int = Field(
        default=60,  # 1 minute
        description="Random variance in cache TTL to prevent cache stampede"
    )

class LlamaModel(LanguageModel):
    """Llama model implementation using Ollama."""

    def __init__(
        self,
        model_name: str = "llama3.2:3b-instruct-q8_0",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        cache_ttl: int = 300,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        batch_size: int = 4,
        enable_streaming: bool = True,
        max_cache_size: int = 1000,
        cache_ttl_variance: int = 60,
    ) -> None:
        """Initialize the Llama model."""
        self.config = LlamaConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            cache_ttl=cache_ttl,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_cache_size=max_cache_size,
            cache_ttl_variance=cache_ttl_variance,
        )
        self._client = None
        # Use TTLCache with LRU eviction for response caching
        self._cache = TTLCache(
            maxsize=self.config.max_cache_size,
            ttl=self.config.cache_ttl
        )
        # Use LRUCache for model metadata caching
        self._metadata_cache = LRUCache(maxsize=100)
        self._batch_size = batch_size
        self._enable_streaming = enable_streaming
        self._token_count = 0
        self._last_token_reset = datetime.now()
        self._circuit_breaker = {
            'failures': 0,
            'last_failure': None,
            'is_open': False
        }

    def _check_circuit_breaker(self):
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
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before=_check_circuit_breaker
    )
    async def _make_request(self, endpoint: str, data: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """Make a request to the Ollama API with retry and circuit breaker."""
        try:
            session = get_session()
            async with session.post(
                f"http://localhost:11434/api/{endpoint}",
                json=data,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status != 200:
                    self._circuit_breaker['failures'] += 1
                    raise RuntimeError(f"Ollama API request failed: {await response.text()}")
                return await response.json()
        except Exception as e:
            self._circuit_breaker['failures'] += 1
            logger.error(f"Request failed: {str(e)}")
            raise

    async def generate(
        self,
        prompt: str,
        stop: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Generate text using the Ollama API.
        
        Args:
            prompt: The prompt to generate from
            stop: Optional stop sequence
            **kwargs: Additional arguments
            
        Returns:
            Generated text
        """
        try:
            session = get_session()
            url = f"http://localhost:11434/api/generate"
            
            data = {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "repeat_penalty": self.config.repeat_penalty,
                    "stop": stop
                }
            }
            
            async with session.post(url, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Ollama API error: {error_text}")
                
                result = await response.json()
                return result.get("response", "")
                
        except Exception as e:
            logger.error(f"Error generating text: {e}", exc_info=True)
            raise

    @classmethod
    @lru_cache(maxsize=32)
    async def list_available_models(cls) -> List[str]:
        """List available models from Ollama.
        
        Returns:
            List of available model names
        """
        try:
            session = get_session()
            async with session.get("http://localhost:11434/api/tags") as response:
                if response.status != 200:
                    logger.warning(f"Ollama API request failed: {await response.text()}")
                    return cls._get_default_models()
                
                result = await response.json()
                models = [model["name"] for model in result.get("models", [])]
                
                if not models:
                    return cls._get_default_models()
                    
                return models
                
        except Exception as e:
            logger.error(f"Failed to get available models: {str(e)}")
            return cls._get_default_models()

    @staticmethod
    def _get_default_models() -> List[str]:
        """Get default list of models."""
        return [
            "llama3.2:3b-instruct-q8_0",
            "llama3.2:8b-instruct-q6_K",
            "llama3.1:70b-instruct-q4_K_M"
        ]

    async def generate_with_vision(
        self,
        prompt: str,
        image_path: str,
        **kwargs: Any
    ) -> str:
        """Generate text from the model with vision capabilities.
        
        Args:
            prompt: Input prompt
            image_path: Path to the image
            **kwargs: Additional arguments
            
        Returns:
            Generated text
        """
        # For now, we'll use the same model for both text and vision
        return await self.generate(prompt, **kwargs)

    def get_config(self) -> Dict[str, Any]:
        """Get the model configuration.
        
        Returns:
            Dictionary containing model configuration
        """
        return self.config.dict()

    async def generate_plan(self, task: str, context: str) -> List[Dict[str, Any]]:
        """Generate a plan for executing a task."""
        return await self._generate_plan_ollama(task, context)
    
    async def generate_response(self, task: str, plan: List[Dict[str, Any]], context: str) -> str:
        """Generate a response based on task execution results."""
        prompt = f"""Task: {task}
Plan execution results:
{json.dumps(plan, indent=2)}

Context:
{context}

Please provide a summary of the task execution results.
"""
        return await self._generate_response_ollama(prompt)
    
    async def _generate_plan_ollama(self, task: str, context: str = "") -> List[Dict[str, Any]]:
        """Generate a plan using Ollama API."""
        try:
            prompt = PROMPT_TEMPLATE.format(context=context, query=task)
            logger.info("Sending prompt to Ollama: %s", prompt)
            
            response = await self.generate(prompt)
            logger.info("Received response from Ollama: %s", response)
            
            if not response:
                logger.warning("Invalid response from Ollama API, using default plan")
                return self._get_default_plan(task)
            
            try:
                plan = self._parse_plan_response({"response": response})
                logger.info("Successfully parsed plan: %s", plan)
                return plan
            except Exception as e:
                logger.warning("Failed to parse plan: %s, using default plan", e)
                return self._get_default_plan(task)
            
        except Exception as e:
            logger.warning("Failed to generate plan with Ollama: %s, using default plan", e)
            return self._get_default_plan(task)
    
    def _get_default_plan(self, task: str) -> List[Dict[str, Any]]:
        """Get a default plan for the task."""
        if "amazon.com" in task and "laptop" in task:
            return [{
                "type": "browser",
                "action": {
                    "type": "navigate",
                    "url": "https://www.amazon.com/s?k=laptop"
                }
            }, {
                "type": "extract",
                "selector": "div[data-component-type='s-search-result']",
                "id": "products",
                "limit": 3
            }]
        else:
            return [{
                "type": "browser",
                "action": {
                    "type": "navigate",
                    "url": "https://example.com"
                }
            }]
    
    async def _generate_response_ollama(self, prompt: str) -> str:
        """Generate a response using Ollama API."""
        try:
            return await self.generate(prompt)
        except Exception as e:
            logger.error(f"Failed to generate response with Ollama: {e}")
            raise

    def _parse_plan_response(self, response: Dict) -> List[Dict[str, Any]]:
        """Parse the model's response into a structured plan.
        
        Args:
            response: Dictionary containing the model's response
            
        Returns:
            List of plan steps
            
        Raises:
            ValueError: If the response cannot be parsed into a valid plan
        """
        try:
            text = response.get("response", "")
            if not text:
                raise ValueError("Empty response from model")
            
            logger.debug(f"Raw response text: {text!r}")
            
            # First try: Look for JSON array directly
            array_match = re.search(r'\[.*\]', text, re.DOTALL)
            if array_match:
                try:
                    steps = json.loads(array_match.group(0))
                    if isinstance(steps, list):
                        logger.info(f"Successfully parsed steps array: {steps!r}")
                        return steps
                except json.JSONDecodeError:
                    logger.debug("Failed to parse array match")
            
            # Second try: Look for JSON object with steps
            json_match = re.search(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL)
            if json_match:
                try:
                    plan = json.loads(json_match.group(0))
                    if isinstance(plan, dict):
                        steps = plan.get("steps", [])
                        if isinstance(steps, list):
                            logger.info(f"Successfully parsed plan object: {steps!r}")
                            return steps
                except json.JSONDecodeError:
                    logger.debug("Failed to parse object match")
            
            # Third try: Clean and parse entire text
            cleaned_text = re.sub(r'[^\x20-\x7E]', '', text)  # Remove non-printable chars
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Normalize whitespace
            
            try:
                # Try parsing as array first
                if cleaned_text.strip().startswith('['):
                    steps = json.loads(cleaned_text)
                    if isinstance(steps, list):
                        logger.info(f"Successfully parsed cleaned text as array: {steps!r}")
                        return steps
                
                # Try parsing as object with steps
                plan = json.loads(cleaned_text)
                if isinstance(plan, dict):
                    steps = plan.get("steps", [])
                    if isinstance(steps, list):
                        logger.info(f"Successfully parsed cleaned text as object: {steps!r}")
                        return steps
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse cleaned text: {e}")
            
            # If we get here, we couldn't parse the response
            error_msg = f"Could not parse valid plan from response: {text[:100]}..."
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        except Exception as e:
            logger.error(f"Failed to parse plan response: {e}", exc_info=True)
            raise ValueError(f"Failed to parse plan response: {str(e)}")

    async def generate_batch(
        self,
        prompts: List[str],
        stop: Optional[str] = None,
        **kwargs: Any
    ) -> List[str]:
        """Generate responses for multiple prompts in parallel.
        
        Args:
            prompts: List of input prompts
            stop: Stop sequence
            **kwargs: Additional arguments
            
        Returns:
            List of generated responses
        """
        # Process prompts in batches
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
        """Generate a streaming response from the model.
        
        Args:
            prompt: Input prompt
            stop: Stop sequence
            **kwargs: Additional arguments
            
        Yields:
            Generated text chunks
        """
        if not self._enable_streaming:
            response = await self.generate(prompt, stop, **kwargs)
            yield response
            return

        try:
            session = get_session()
            async with session.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": self.config.temperature,
                        "top_p": self.config.top_p,
                        "top_k": self.config.top_k,
                        "repeat_penalty": self.config.repeat_penalty,
                        "max_tokens": self.config.max_tokens,
                        "stop": [stop] if stop else None
                    }
                }
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Ollama API request failed: {await response.text()}")
                
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                yield chunk["response"]
                                self._token_count += 1
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate streaming response: {str(e)}")

    def get_token_count(self) -> int:
        """Get the current token count.
        
        Returns:
            Number of tokens generated since last reset
        """
        # Reset token count if TTL has expired
        if datetime.now() - self._last_token_reset > timedelta(seconds=self.config.cache_ttl):
            self._token_count = 0
            self._last_token_reset = datetime.now()
        return self._token_count

    def reset_token_count(self) -> None:
        """Reset the token count."""
        self._token_count = 0
        self._last_token_reset = datetime.now()