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
import argparse # Import argparse

# Configure basic logging
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NovaRun")

# Import necessary Nova components AFTER loading .env
from nova.core.llm.llm import LLMConfig
from nova.core.browser import BrowserConfig
from nova.core.agents import TaskAgent

async def main():
    # Load environment variables
    load_dotenv()
    
    # Setup argument parsing
    parser = argparse.ArgumentParser(description="Run the Nova agent with a specified task.")
    parser.add_argument(
        "task", 
        type=str, 
        nargs='?', # Make the task argument optional
        default="Navigate to example.com, find the main heading, and tell me what it is.", # Default task
        help="The task description for the Nova agent to perform."
    )
    args = parser.parse_args()
    task = args.task

    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Define the task for the agent (Now comes from args)
    # task = "Navigate to example.com, find the main heading, and tell me what it is."

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
         logger.error("Make sure all dependencies are installed correctly (`pip install -e \".[dev]\") including `langchain-nvidia-ai-endpoints`.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}") # Log full traceback
    finally:
        # Ensure proper cleanup
        if 'agent' in locals() and hasattr(agent, 'stop'): # Check if stop method exists
            logger.info("Attempting agent cleanup...")
            await agent.stop()
            logger.info("Agent cleanup finished.")
        else:
            logger.warning("Agent object not found or does not have a stop method for cleanup.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Task interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)