"""Extended agent implementation with higher-level functionality."""

from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from nova.core.agent import Agent as CoreAgent
from nova.core.config import AgentConfig, BrowserConfig, LLMConfig
from nova.core.llm import LLM
from nova.core.memory import Memory
from nova.core.tools import Tool


class Agent(CoreAgent):
    """Extended agent with additional functionality.
    
    This agent provides a higher-level interface for browser automation and task execution,
    with built-in support for LLM-based decision making and memory management.
    """

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[Tool]] = None,
        memory: Optional[Memory] = None,
        config: Optional[AgentConfig] = None,
        browser_config: Optional[BrowserConfig] = None,
    ) -> None:
        """Initialize the agent with optional components.
        
        Args:
            llm: Language model for decision making. If None, uses default NIMProvider.
            tools: List of tools available to the agent.
            memory: Memory system for context management.
            config: Agent configuration.
            browser_config: Browser configuration.
        """
        # Initialize core agent
        super().__init__(
            llm=LLM(
                provider=config.llm_config.provider if config else "nim",
                docker_image=config.llm_config.nim_config.docker_image if config else "nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1:latest",
                api_base=config.llm_config.nim_config.api_base if config else "http://localhost:8000",
                model_name=config.llm_config.model_name if config else "nvidia/llama-3.3-nemotron-super-49b-v1",
                temperature=config.llm_config.temperature if config else 0.2,
                max_tokens=config.llm_config.max_tokens if config else 4096,
                batch_size=config.llm_config.batch_size if config else 4,
                enable_streaming=config.llm_config.enable_streaming if config else True,
            ),
            tools=tools,
            memory=memory,
            config=config,
            browser_config=browser_config,
        ) 