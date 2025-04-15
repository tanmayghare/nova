"""Configuration management for Nova."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import os

class BrowserConfig(BaseModel):
    """Configuration for browser automation."""

    headless: bool = Field(
        default=True,
        description="Whether to run in headless mode"
    )
    action_timeout: float = Field(
        default=10.0,
        description="Default timeout for actions like click/type in seconds"
    )
    navigation_timeout: float = Field(
        default=30.0,
        description="Default timeout for page navigation in seconds"
    )
    wait_timeout: float = Field(
        default=10.0,
        description="Default timeout for explicit waits in seconds"
    )
    viewport: Dict[str, int] = Field(
        default={
            "width": int(os.environ.get("BROWSER_VIEWPORT_WIDTH", "1280")),
            "height": int(os.environ.get("BROWSER_VIEWPORT_HEIGHT", "720"))
        },
        description="Viewport dimensions"
    )
    browser_args: List[str] = Field(
        default=[],
        description="Additional arguments to pass to the browser launch command"
    )
    user_agent: Optional[str] = Field(
        default=None,
        description="Custom user agent string"
    )


class NIMConfig(BaseModel):
    """Configuration for NVIDIA NIM."""
    docker_image: str = Field(
        default=os.environ.get("NIM_DOCKER_IMAGE"),
        description="Docker image for NIM service"
    )
    api_base: str = Field(
        default=os.environ.get("NIM_API_BASE_URL"),
        description="Base URL for NIM API"
    )
    api_key: Optional[str] = Field(
        default=os.environ.get("NIM_API_KEY"),
        description="API key for NIM service"
    )


class LLMConfig(BaseModel):
    """Configuration for language models."""
    provider: str = Field(
        default="nim",
        description="LLM provider to use (nim or ollama)"
    )
    model_name: str = Field(
        default=os.environ.get("MODEL_NAME"),
        description="Name of the model to use"
    )
    temperature: float = Field(
        default=float(os.environ.get("MODEL_TEMPERATURE", 0.1)),
        description="Temperature for sampling"
    )
    max_tokens: int = Field(
        default=int(os.environ.get("MODEL_MAX_TOKENS", 1500)),
        description="Maximum number of tokens to generate"
    )
    batch_size: int = Field(
        default=int(os.environ.get("MODEL_BATCH_SIZE", 1)),
        description="Maximum number of requests to process in parallel"
    )
    enable_streaming: bool = Field(
        default=os.environ.get("MODEL_ENABLE_STREAMING", "False").lower() == "true",
        description="Whether to enable response streaming"
    )
    nim_config: NIMConfig = Field(
        default_factory=NIMConfig,
        description="NVIDIA NIM specific configuration"
    )


class AgentConfig(BaseModel):
    """Configuration for agents."""
    name: str = Field(
        default="Nova",
        description="Name of the agent"
    )
    llm_config: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration"
    )
    tools: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool configurations"
    )
    memory: Dict[str, Any] = Field(
        default_factory=dict,
        description="Memory configurations"
    )
    base_prompt: str = Field(
        default="You are an autonomous agent designed to follow instructions and complete tasks by thinking step-by-step and using the available tools.",
        description="Base system prompt for the agent's LLM."
    )

    max_steps: int = 100
    max_failures: int = 3
    retry_delay: int = 5
    use_vision: bool = True
    enable_memory: bool = True
    max_iterations: int = 10
    max_history_context_iterations: int = 5
    confidence_threshold: float = 0.7
    max_retries: int = 3
    retry_delay: float = 2.0
    max_tool_errors: int = 3

    class Config:
        extra = "allow" 