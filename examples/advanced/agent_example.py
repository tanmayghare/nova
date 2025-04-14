import asyncio
import os
from typing import Any, Dict, Sequence

from nova.core.llama import LlamaModel
from nova.core.agent import Agent
from nova.core.config import AgentConfig, BrowserConfig
from nova.core.llm import LLM
from nova.core.memory import Memory
from nova.core.tools import Tool
from nova.tools.base.tool import ToolResult
try:
    from nova.learning.interaction_logger import InteractionLogger
except ImportError:
    InteractionLogger = None
    print("Warning: InteractionLogger not found. Interaction logging will be disabled.")


class SearchTool(Tool):
    """A tool that performs web searches."""

    def __init__(self) -> None:
        super().__init__("search", "Performs a web search")

    async def execute(self, input_data: Dict[str, Any]) -> str:
        query = input_data.get("query", "")
        # In a real implementation, this would call a search API
        result_str = f"Search results for: {query}"
        # Return a ToolResult object
        return ToolResult(success=True, data=result_str)


class CalculatorTool(Tool):
    """A tool that performs calculations."""

    def __init__(self) -> None:
        super().__init__("calculate", "Performs mathematical calculations")

    async def execute(self, input_data: Dict[str, Any]) -> str:
        expression = input_data.get("expression", "")
        try:
            result = eval(expression)  # In a real implementation, use a safer evaluation method
            # Return a ToolResult object
            return ToolResult(success=True, data=str(result))
        except Exception as e:
            # Return a ToolResult object for errors too
            return ToolResult(success=False, error=f"Error calculating: {str(e)}")


async def main() -> None:
    # Initialize components
    # Explicitly create the LLM instance configured for the default NIM provider
    # providing the required default configuration values.
    llm = LLM(
        provider="nim",
        docker_image="nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1:latest",
        api_base="http://localhost:8000",
        model_name="nvidia/llama-3.3-nemotron-super-49b-v1"
        # Use other LLM defaults like batch_size, enable_streaming implicitly
    )
    memory = Memory()
    tools: Sequence[Tool] = [SearchTool(), CalculatorTool()]
    
    # Setup Interaction Logger
    logger_instance = None
    if InteractionLogger:
        log_dir = "logs"
        log_file = os.path.join(log_dir, "interaction_log.jsonl")
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        logger_instance = InteractionLogger(log_file)
        print(f"Interaction logging enabled to: {log_file}")
    
    # Create configurations
    agent_config = AgentConfig() # Uses default LLMConfig internally if needed elsewhere
    browser_config = BrowserConfig(headless=True)
    
    # Create agent - Pass the explicitly created default LLM AND logger
    agent = Agent(
        llm=llm, # Pass the default NIM LLM instance
        tools=tools,
        memory=memory,
        config=agent_config,
        browser_config=browser_config,
        interaction_logger=logger_instance # <-- Pass logger instance
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
