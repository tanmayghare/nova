"""Basic example of using Nova with LangChain."""

import os
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentExample")

# Import Nova components after loading .env
try:
    from nova.core.llm import LLM, LLMConfig
    from nova.core.browser import Browser
    from nova.core.memory import Memory
    from nova.core.agent.langchain_agent import LangChainAgent
except ImportError as e:
    logger.error(f"Import Error: {e}")
    logger.error("Make sure Nova is installed correctly (`pip install -e .[dev]`) and dependencies are available.")
    exit(1)

async def main():
    """Runs a simple task using the LangChainAgent."""
    # Define a simple task for the agent
    task = "Navigate to duckduckgo.com and search for 'large language models'."
    logger.info(f"Starting LangChain Agent for task: \"{task}\"")

    agent = None
    try:
        # Initialize configurations
        llm_config = LLMConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1000
        )
        llm = LLM(config=llm_config)
        
        # Initialize browser and memory
        browser = Browser()
        memory = Memory()

        # Create the LangChain Agent
        agent = LangChainAgent(
            llm=llm,
            browser=browser,
            memory=memory
        )

        # Run the task
        logger.info("Running agent...")
        result = await agent.run(task)

        # Print the final result
        logger.info("--- Task Execution Result ---")
        logger.info(f"Status: {result['status']}")
        if result['status'] == 'success':
            logger.info(f"Result: {result['result']}")
        else:
            logger.error(f"Error: {result['error']}")
        logger.info("--- End of Task ---")

    except ValueError as e:
         logger.error(f"Configuration Error: {e}")
         logger.error("Please ensure your .env file is set up correctly (e.g., API keys, model name).")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the task: {e}")
    finally:
        if agent and agent.browser:
            logger.info("Stopping browser...")
            await agent.browser.stop()
            logger.info("Browser stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Task interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error running example: {e}", exc_info=True) 