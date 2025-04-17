# examples/basic/agent_example.py
import asyncio
import logging
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file first
load_dotenv()

# Configure basic logging
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentExample")

# Import Nova components after loading .env
try:
    from nova.core.llm.llm import LLMConfig
    from nova.core.browser import BrowserConfig
    from nova.agents.task.task_agent import TaskAgent
except ImportError as e:
    logger.error(f"Import Error: {e}")
    logger.error("Make sure Nova is installed correctly (`pip install -e .[dev]`) and dependencies are available.")
    exit(1)

async def main():
    """Runs a simple task using the TaskAgent."""
    # Define a simple task for the agent
    task = "Navigate to duckduckgo.com and search for 'large language models'."
    logger.info(f"Starting Task Agent for task: \"{task}\"")

    agent = None # Initialize agent to None for finally block
    try:
        # Initialize configurations (will read from .env)
        llm_config = LLMConfig()
        browser_config = BrowserConfig()

        logger.info(f"Using LLM Provider: {llm_config.primary_provider}")
        logger.info(f"Using LLM Model: {llm_config.primary_model}")
        logger.info(f"Browser Headless Mode: {browser_config.headless}")

        # Create the Task Agent
        agent = TaskAgent(
            task_id="basic-example-001",
            task_description=task,
            llm_config=llm_config,
            browser_config=browser_config
        )

        # Run the task
        logger.info("Running agent...")
        result = await agent.run()

        # Print the final result
        logger.info("--- Task Execution Result ---")
        print(json.dumps(result.dict(), indent=2))
        logger.info("--- End of Task ---")

    except ValueError as e:
         logger.error(f"Configuration Error: {e}")
         logger.error("Please ensure your .env file is set up correctly (e.g., API keys, model name).")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the task: {e}")
    finally:
        # Ensure proper cleanup if the agent was initialized
        if agent:
            logger.info("Stopping agent...")
            await agent.stop()
            logger.info("Agent stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Task interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error running example: {e}", exc_info=True) 