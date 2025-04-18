"""Configuration classes for Nova."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

@dataclass
class BrowserConfig:
    """Browser configuration."""
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    timeout: float = 15.0
    retry_attempts: int = 3
    proxy: Optional[str] = None
    cookies: Optional[Dict[str, str]] = None
    extra_args: List[str] = field(default_factory=list)

@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: str = "nvidia"
    model_name: str = "meta/llama-3.3-70b-instruct"
    temperature: float = 0.1
    max_tokens: int = 4096
    streaming: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: float = 15.0
    max_retries: int = 3

@dataclass
class MemoryConfig:
    """Memory configuration."""
    max_examples: int = 1000
    similarity_threshold: float = 0.7
    vector_store_path: str = "data/vector_store"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_device: str = "cpu"

@dataclass
class AgentConfig:
    """Agent configuration."""
    max_iterations: int = 10
    max_consecutive_failures: int = 3
    max_parallel_tasks: int = 3
    browser_pool_size: int = 5
    log_level: str = "INFO"
    log_dir: str = "logs"
    cache_dir: str = "cache"
    debug_mode: bool = False
    browser_config: BrowserConfig = field(default_factory=BrowserConfig)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)

    def __post_init__(self):
        """Create required directories."""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.memory_config.vector_store_path).mkdir(parents=True, exist_ok=True)

@dataclass
class ToolConfig:
    """Tool configuration."""
    name: str
    description: str
    version: str = "1.0.0"
    enabled: bool = True
    timeout: float = 15.0
    retry_attempts: int = 3
    dependencies: List[str] = field(default_factory=list)
    extra_config: Dict[str, Any] = field(default_factory=dict) 