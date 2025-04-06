import asyncio
from typing import Any, Dict, List, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from nova.core.agent import Agent
from nova.core.browser import Browser
from nova.core.config import AgentConfig, BrowserConfig
from nova.core.llm import LLM
from nova.core.memory import Memory
from nova.core.tools import Tool, ToolRegistry


class SearchTool(Tool):
    """A tool that performs web searches."""

    def __init__(self) -> None:
        super().__init__("search", "Performs a web search")

    async def execute(self, input_data: Dict[str, Any]) -> str:
        query = input_data.get("query", "")
        # In a real implementation, this would call a search API
        return f"Search results for: {query}"


class CalculatorTool(Tool):
    """A tool that performs calculations."""

    def __init__(self) -> None:
        super().__init__("calculate", "Performs mathematical calculations")

    async def execute(self, input_data: Dict[str, Any]) -> str:
        expression = input_data.get("expression", "")
        # In a real implementation, this would evaluate the expression
        return f"Result of {expression}"


async def main() -> None:
    # Initialize components
    model = ChatOpenAI()  # Configure with your API key
    llm = LLM(model)
    memory = Memory()
    tools: Sequence[Tool] = [SearchTool(), CalculatorTool()]
    
    # Create configurations
    agent_config = AgentConfig()
    browser_config = BrowserConfig(headless=True)
    
    # Create agent
    agent = Agent(
        llm=llm,
        tools=tools,
        memory=memory,
        config=agent_config,
        browser_config=browser_config,
    )
    
    # Example tasks
    tasks = [
        "Search for information about Python web frameworks",
        "Calculate 2 + 2 * 3",
        "Find the latest news about AI",
    ]
    
    # Run tasks
    for task in tasks:
        print(f"\nExecuting task: {task}")
        result = await agent.run(task)
        print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
