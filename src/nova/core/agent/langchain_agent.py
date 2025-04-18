"""LangChain-based agent implementation."""

from typing import Any, Dict, List, Optional
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from ..llm import LLM
from ..memory import Memory
from ..browser import Browser
from nova.tools.browser import get_browser_tools

class LangChainAgent:
    """Agent implementation using LangChain's AgentExecutor."""
    
    def __init__(
        self,
        llm: LLM,
        browser: Optional[Browser] = None,
        memory: Optional[Memory] = None,
        tools: Optional[List[BaseTool]] = None,
        system_message: Optional[str] = None
    ):
        """Initialize the LangChain agent.
        
        Args:
            llm: LLM instance to use
            browser: Optional browser instance for browser tools
            memory: Optional memory instance
            tools: Optional list of additional tools
            system_message: Optional custom system message
        """
        self.llm = llm
        self.browser = browser
        self.memory = memory or Memory()
        
        # Get base tools
        self.tools: List[BaseTool] = []
        
        # Add browser tools if browser is provided
        if browser:
            self.tools.extend(get_browser_tools(browser))
            
        # Add any additional tools
        if tools:
            self.tools.extend(tools)
            
        # Create the agent
        self.agent = self._create_agent(system_message)
        
    def _create_agent(self, system_message: Optional[str] = None) -> AgentExecutor:
        """Create the LangChain agent executor."""
        # Default system message
        if not system_message:
            system_message = """You are Nova, an AI assistant specialized in web automation tasks.
            You have access to browser tools for navigation and interaction.
            Use these tools to help users accomplish their tasks on the web."""
            
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        agent = create_openai_tools_agent(
            llm=self.llm.get_langchain_llm(),
            tools=self.tools,
            prompt=prompt
        )
        
        # Create the executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
        
    async def run(self, task: str) -> Dict[str, Any]:
        """Run the agent on a task.
        
        Args:
            task: The task description to execute
            
        Returns:
            Dictionary containing the execution results
        """
        try:
            # Start browser if available
            if self.browser:
                await self.browser.start()
                
            # Run the agent
            result = await self.agent.ainvoke({"input": task})
            
            return {
                "status": "success",
                "result": result.get("output", "Task completed."),
                "error": None
            }
            
        except Exception as e:
            return {
                "status": "error",
                "result": None,
                "error": str(e)
            }
            
        finally:
            # Clean up
            if self.browser:
                await self.browser.stop() 