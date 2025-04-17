"""Task Agent Configuration."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class TaskAgentConfig:
    """Configuration for the TaskAgent."""
    
    # Task execution settings
    max_retries: int = 3
    timeout_seconds: int = 300
    
    # LLM settings
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.1
    max_tokens: int = 1000
    
    # Browser settings
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 800
    
    # Memory settings
    max_memory_items: int = 100
    memory_ttl_seconds: int = 3600 