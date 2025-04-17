"""Plan-and-Execute Agent Implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool
from langchain_core.output_parsers import JsonOutputParser

from ...core.llm import LLM, LLMConfig
from ...core.memory import Memory
from ...core.tools import Tool, ToolRegistry
from ...core.browser import Browser, BrowserConfig
from ..base.base_agent import BaseAgent, AgentState
from .config import PlanExecuteAgentConfig

logger = logging.getLogger(__name__)

class PlanExecuteAgent(BaseAgent):
    """Agent implementing the Plan-and-Execute pattern.
    
    This agent:
    1. Creates a detailed plan for task execution
    2. Executes each step of the plan
    3. Monitors progress and adjusts the plan if needed
    4. Reports results and learns from experience
    """
    
    def __init__(
        self,
        llm_config: LLMConfig,
        browser_config: Optional[BrowserConfig] = None,
        memory: Optional[Memory] = None,
        tools: Optional[List[Tool]] = None,
        plan_execute_config: Optional[PlanExecuteAgentConfig] = None,
    ):
        """Initialize the Plan-and-Execute agent."""
        super().__init__(
            llm_config=llm_config,
            browser_config=browser_config,
            memory=memory,
            tools=tools,
        )
        self._state = AgentState()
        self._planner_executor: Optional[AgentExecutor] = None
        self._plan_execute_config = plan_execute_config or PlanExecuteAgentConfig()
        
    async def _initialize_agent(self) -> None:
        """Initialize the Plan-and-Execute agent components."""
        if self._planner_executor is not None:
            return
            
        # Convert our tools to LangChain tools
        tools = self.tool_registry.get_tools()
        if self._plan_execute_config.allowed_tools:
            tools = [
                tool for tool in tools
                if tool.name in self._plan_execute_config.allowed_tools
            ]
            
        langchain_tools = [
            BaseTool(
                name=tool.name,
                description=tool.description,
                func=tool.execute,
            )
            for tool in tools
        ]
        
        # Create the planner prompt
        planner_prompt = ChatPromptTemplate.from_messages([
            ("system", self._plan_execute_config.planner_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the executor prompt
        executor_prompt = ChatPromptTemplate.from_messages([
            ("system", self._plan_execute_config.executor_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the planner agent
        planner_agent = create_react_agent(
            llm=self.llm,
            tools=langchain_tools,
            prompt=planner_prompt,
        )
        
        # Create the executor agent
        executor_agent = create_react_agent(
            llm=self.llm,
            tools=langchain_tools,
            prompt=executor_prompt,
        )
        
        # Create the combined agent executor
        self._planner_executor = AgentExecutor(
            agent=planner_agent,
            tools=langchain_tools,
            verbose=self._plan_execute_config.verbose,
            handle_parsing_errors=self._plan_execute_config.handle_parsing_errors,
            max_iterations=self._plan_execute_config.max_iterations,
        )
        
    async def _create_plan(self, task: str) -> Dict[str, Any]:
        """Create an execution plan for the given task."""
        try:
            result = await self._planner_executor.ainvoke({
                "input": f"Create a detailed plan for: {task}",
                "chat_history": self.memory.get_history() if self.memory else [],
            })
            return result
        except Exception as e:
            logger.error(f"Plan creation failed: {str(e)}")
            raise
            
    async def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the given plan."""
        try:
            result = await self._planner_executor.ainvoke({
                "input": f"Execute the following plan: {plan}",
                "chat_history": self.memory.get_history() if self.memory else [],
            })
            return result
        except Exception as e:
            logger.error(f"Plan execution failed: {str(e)}")
            raise
            
    async def run(self, task: str) -> Dict[str, Any]:
        """Run a task using the Plan-and-Execute pattern.
        
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
            self._state.status = "planning"
            
            # Create the plan
            plan = await self._create_plan(task)
            
            # Update state
            self._state.status = "executing"
            self._state.plan = plan
            
            # Execute the plan
            result = await self._execute_plan(plan)
            
            # Update state
            self._state.status = "completed"
            self._state.result = result
            
            return {
                "status": "success",
                "plan": plan,
                "result": result,
                "error": None,
            }
            
        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}")
            self._state.status = "failed"
            self._state.error = str(e)
            
            return {
                "status": "failed",
                "plan": self._state.plan if hasattr(self._state, "plan") else None,
                "result": None,
                "error": str(e),
            }
            
    async def cleanup(self) -> None:
        """Clean up resources."""
        await super().cleanup()
        self._planner_executor = None 