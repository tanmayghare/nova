import os
import asyncio
import logging
import json
from typing import Any, Dict, Sequence

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    llm = LLM(
        provider="nim",
        docker_image=os.environ.get("NIM_DOCKER_IMAGE"),
        api_base=os.environ.get("NIM_API_BASE_URL"),
        model_name=os.environ.get("MODEL_NAME")
    )
    memory = Memory()
    tools: Sequence[Tool] = [SearchTool(), CalculatorTool()]
    
    # Setup Interaction Logger
    interaction_logger = None
    if InteractionLogger:
        log_dir = "logs"
        log_file = os.path.join(log_dir, "interaction_log.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        interaction_logger = InteractionLogger(log_file)
        logger.info(f"Interaction logging enabled to: {log_file}")
    
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
        interaction_logger=interaction_logger
    )
    
    try:
        # Start the agent and browser pool
        await agent.start()
        
        # Example tasks
        tasks = [
            "Search for information about Python web frameworks",
            "Calculate 2 + 2 * 3",
            "Find the latest news about AI",
        ]
        
        # Add complex NLP test task
        tasks.append({
            'name': 'Complex NLP Test',
            'description': 'Navigate to GitHub, search for "machine learning", click on the first repository, and take a screenshot of the README',
            'type': 'interpreted',
            'expected_result': 'Screenshot of a machine learning repository README'
        })
        
        # Run tasks
        for task in tasks:
            if isinstance(task, dict):
                task_name = task['name']
                task_desc = task['description']
                task_type = task['type']
            else:
                task_name = task
                task_desc = task
                task_type = 'regular'
                
            logger.info(f"\nExecuting task: {task_name}")
            logger.info(f"Description: {task_desc}")
            
            try:
                if task_type == 'interpreted':
                    result = await agent._execute_interpreted_plan(task_desc)
                else:
                    result = await agent.run(task_desc)
                    
                if result is None:
                    logger.info("Task completed successfully with no result!")
                    continue
                    
                logger.info(f"Task result: {json.dumps(result, indent=2)}")
                
                if result.get('status') == 'success':
                    logger.info("Task completed successfully!")
                    if isinstance(task, dict) and task.get('expected_result'):
                        logger.info(f"Expected result: {task['expected_result']}")
                else:
                    logger.error(f"Task failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Error executing task: {e}", exc_info=True)
    finally:
        # Stop the agent and browser pool
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
