"""Mock LLM implementation for testing."""

import json
from typing import Any, Optional
from datetime import datetime, timedelta

class MockLLM:
    """Mock LLM implementation for testing."""
    
    def __init__(
        self,
        model_name: str = "mock-model",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        cache_ttl: int = 300,
        max_retries: int = 3
    ):
        """Initialize the mock LLM."""
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        
        self._cache = {}
        self._last_cache_cleanup = datetime.now()
        self._calls = 0
        
    async def generate(
        self,
        prompt: str,
        stop: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """Generate text from the model."""
        self._calls += 1
        
        # Check cache
        cache_key = f"{prompt}:{stop}:{json.dumps(kwargs)}"
        if cache_key in self._cache:
            cached_response, timestamp = self._cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return cached_response
                
        # Clean up old cache entries
        self._cleanup_cache()
        
        # Generate mock response based on prompt
        if "amazon" in prompt.lower() and "laptop" in prompt.lower():
            response = {
                "steps": [
                    {
                        "type": "browser",
                        "action": {
                            "type": "navigate",
                            "url": "https://www.amazon.com/s?k=gaming+laptop"
                        }
                    },
                    {
                        "type": "extract",
                        "selector": "div[data-component-type='s-search-result']",
                        "id": "products",
                        "limit": 3
                    }
                ]
            }
        else:
            response = {
                "steps": [
                    {
                        "type": "browser",
                        "action": {
                            "type": "navigate",
                            "url": "https://example.com"
                        }
                    }
                ]
            }
            
        response_text = json.dumps(response)
        
        # Cache the response
        self._cache[cache_key] = (response_text, datetime.now())
        
        return response_text
        
    def _cleanup_cache(self) -> None:
        """Clean up old cache entries."""
        now = datetime.now()
        if now - self._last_cache_cleanup > timedelta(seconds=self.cache_ttl):
            self._cache = {
                k: v for k, v in self._cache.items()
                if now - v[1] < timedelta(seconds=self.cache_ttl)
            }
            self._last_cache_cleanup = now
            
    def get_calls(self) -> int:
        """Get the number of API calls made."""
        return self._calls
        
    def reset_calls(self) -> None:
        """Reset the call counter."""
        self._calls = 0 