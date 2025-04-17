"""ReAct Agent Implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from ...core.llm import LLM, LLMConfig
from ...core.memory import Memory
from ...core.tools import Tool, ToolRegistry
from ...core.browser import Browser, BrowserConfig
from ..base.base_agent import BaseAgent, AgentState
from .config import ReActAgentConfig

logger = logging.getLogger(__name__)

class ReActAgent(BaseAgent):
    """Agent implementing the ReAct (Reasoning and Acting) pattern.
    
    This agent uses LangChain's ReAct implementation to:
    1. Reason about the current state
    2. Take actions based on reasoning
    3. Observe the results
    4. Repeat until task completion
    """
    
    def __init__(
        self,
        llm_config: LLMConfig,
        browser_config: Optional[BrowserConfig] = None,
        memory: Optional[Memory] = None,
        tools: Optional[List[Tool]] = None,
        react_config: Optional[ReActAgentConfig] = None,
    ):
        """Initialize the ReAct agent."""
        super().__init__(
            llm_config=llm_config,
            browser_config=browser_config,
            memory=memory,
            tools=tools,
        )
        self._state = AgentState()
        self._agent_executor: Optional[AgentExecutor] = None
        self._react_config = react_config or ReActAgentConfig()
        
    async def _initialize_agent(self) -> None:
        """Initialize the ReAct agent executor."""
        if self._agent_executor is not None:
            return
            
        # Convert our tools to LangChain tools
        tools = self.tool_registry.get_tools()
        if self._react_config.allowed_tools:
            tools = [
                tool for tool in tools
                if tool.name in self._react_config.allowed_tools
            ]
            
        langchain_tools = [
            BaseTool(
                name=tool.name,
                description=tool.description,
                func=tool.execute,
            )
            for tool in tools
        ]
        
        # Create the ReAct prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._react_config.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=langchain_tools,
            prompt=prompt,
        )
        
        # Create the agent executor
        self._agent_executor = AgentExecutor(
            agent=agent,
            tools=langchain_tools,
            verbose=self._react_config.verbose,
            handle_parsing_errors=self._react_config.handle_parsing_errors,
            max_iterations=self._react_config.max_iterations,
        )
        
    async def run(self, task: str) -> Dict[str, Any]:
        """Run a task using the ReAct pattern.
        
        Args:
            task: The task description to execute
            
        Returns:
            Dictionary containing the execution results
        """
        try:
            # Initialize the agent if needed
            await self._initialize_agent()
            
            # Update state
            self._state.task = task
            self._state.status = "running"
            
            # Execute the task
            result = await self._agent_executor.ainvoke({
                "input": task,
                "chat_history": self.memory.get_history() if self.memory else [],
            })
            
            # Update state
            self._state.status = "completed"
            self._state.result = result
            
            return {
                "status": "success",
                "result": result,
                "error": None,
            }
            
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            self._state.status = "failed"
            self._state.error = str(e)
            
            return {
                "status": "failed",
                "result": None,
                "error": str(e),
            }
            
    async def cleanup(self) -> None:
        """Clean up resources."""
        await super().cleanup()
        self._agent_executor = None 