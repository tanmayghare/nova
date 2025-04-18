"""Task agent implementation using LangChain."""

from typing import Any, Dict, Optional
from ...core.agent.langchain_agent import LangChainAgent
from ...core.llm import LLM
from ...core.memory import Memory
from ...core.browser import Browser
from ...core.tools.base import BaseTool

class TaskAgent:
    """Agent for executing tasks using LangChain."""
    
    def __init__(
        self,
        llm: LLM,
        browser: Optional[Browser] = None,
        memory: Optional[Memory] = None,
        tools: Optional[list[BaseTool]] = None,
        system_message: Optional[str] = None
    ):
        """Initialize the task agent.
        
        Args:
            llm: LLM instance to use
            browser: Optional browser instance for browser tools
            memory: Optional memory instance
            tools: Optional list of additional tools
            system_message: Optional custom system message
        """
        self.agent = LangChainAgent(
            llm=llm,
            browser=browser,
            memory=memory,
            tools=tools,
            system_message=system_message
        )
        
    async def run(self, task: str) -> Dict[str, Any]:
        """Run the agent on a task.
        
        Args:
            task: The task description to execute
            
        Returns:
            Dictionary containing the execution results
        """
        return await self.agent.run(task) 