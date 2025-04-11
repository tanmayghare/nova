"""Extended agent implementation with higher-level functionality."""

from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from nova.core.agent import Agent as CoreAgent
from nova.core.browser import Browser
from nova.core.config import AgentConfig, BrowserConfig
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
            llm: Language model for decision making. If None, uses default LlamaModel.
            tools: List of tools available to the agent.
            memory: Memory system for context management.
            config: Agent configuration.
            browser_config: Browser configuration.
        """
        # Initialize core agent
        super().__init__(
            llm=LLM(llm),
            tools=tools,
            memory=memory,
            config=config,
            browser_config=browser_config,
        ) 