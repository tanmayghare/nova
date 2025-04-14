"""Configuration management for Nova."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class BrowserConfig(BaseModel):
    """Configuration for browser automation."""

    headless: bool = Field(
        default=True,
        description="Whether to run in headless mode"
    )
    timeout: int = Field(
        default=30,
        description="Default timeout in seconds"
    )
    viewport: Dict[str, int] = Field(
        default={"width": 1280, "height": 720},
        description="Viewport dimensions"
    )
    browser_args: List[str] = []


class NIMConfig(BaseModel):
    """Configuration for NVIDIA NIM."""
    docker_image: str = Field(
        default="nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1:latest",
        description="Docker image for NIM service"
    )
    api_base: str = Field(
        default="http://localhost:8000",
        description="Base URL for NIM API"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for NIM service"
    )


class LLMConfig(BaseModel):
    """Configuration for language models."""
    provider: str = Field(
        default="nim",
        description="LLM provider to use (nim or ollama)"
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
    batch_size: int = Field(
        default=4,
        description="Maximum number of requests to process in parallel"
    )
    enable_streaming: bool = Field(
        default=True,
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

    class Config:
        extra = "allow" 