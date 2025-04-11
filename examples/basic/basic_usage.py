import asyncio
from typing import Any, Dict, Sequence

from nova.core.llama import LlamaModel
from nova.core.agent import Agent
from nova.core.config import AgentConfig, BrowserConfig
from nova.core.llm import LLM
from nova.core.memory import Memory
from nova.core.tools import Tool


class ExampleTool(Tool):
    """An example tool that adds two numbers."""

    def __init__(self) -> None:
        super().__init__("add", "Adds two numbers together")

    async def execute(self, input_data: Dict[str, Any]) -> str:
        a = input_data.get("a", 0)
        b = input_data.get("b", 0)
        return str(a + b)


async def main() -> None:
    # Initialize components
    model = LlamaModel(model_name="llama3.2:3b-instruct-q8_0")
    llm = LLM(model, model_type="llama")
    memory = Memory()
    tools: Sequence[Tool] = [ExampleTool()]
    
    # Create configurations
    agent_config = AgentConfig()
    browser_config = BrowserConfig(headless=True)
    
    # Create and run agent
    agent = Agent(
        llm=llm,
        tools=tools,
        memory=memory,
        config=agent_config,
        browser_config=browser_config,
    )
    
    # Run a task
    result = await agent.run("Add 5 and 7")
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
