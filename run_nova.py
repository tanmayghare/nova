# run_nova.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Enable LangChain debug mode
import langchain
langchain.debug = True

import asyncio
import logging
import sys

# Configure basic logging
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NovaRun")

# Import necessary Nova components AFTER loading .env
from nova.core.llm.llm import LLMConfig
from nova.core.browser import BrowserConfig
from nova.agents.task.task_agent import TaskAgent

async def main():
    # Define the task for the agent
    task = "Navigate to example.com, find the main heading, and tell me what it is."

    logger.info(f"Starting Nova Task Agent for task: \"{task}\"")

    # Initialize configurations (they will read from .env)
    try:
        llm_config = LLMConfig()
        browser_config = BrowserConfig() # Reads BROWSER_* env vars
        logger.info(f"Using LLM Provider: {llm_config.primary_provider}")
        logger.info(f"Using LLM Model: {llm_config.primary_model}")
        logger.info(f"Using NIM Base URL: {llm_config.primary_base_url}")
        logger.info(f"Browser Headless Mode: {browser_config.headless}")

        # Create the Task Agent
        # Memory and Tools are handled internally for basic BrowserTools usage
        agent = TaskAgent(
            task_id="test-run-001",
            task_description=task,
            llm_config=llm_config,
            browser_config=browser_config
            # memory=None, # Uses default Memory
            # tools=None, # Uses default BrowserTools if browser_config provided
        )

        # Run the task
        result = await agent.run()

        # Print the final result
        logger.info("--- Task Execution Result ---")
        # Pretty print the JSON result
        import json
        print(json.dumps(result.dict(), indent=2))
        logger.info("--- End of Task ---")

    except ValueError as e:
         logger.error(f"Configuration Error: {e}")
         logger.error("Please ensure your .env file is set up correctly, especially NIM_API_BASE_URL, MODEL_NAME, and NVIDIA_API_KEY.")
    except ImportError as e:
         logger.error(f"Import Error: {e}")
         logger.error("Make sure all dependencies are installed correctly (`pip install -e \".[dev]\"`) including `langchain-nvidia-ai-endpoints`.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}") # Log full traceback
    finally:
        # Ensure proper cleanup
        if 'agent' in locals():
            await agent.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Task interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
