#!/usr/bin/env python3

import asyncio
import argparse
import logging
import os
from typing import Optional

from nova.core.browser import Browser
from nova.core.llm import LLMConfig
from nova.core.agents import TaskAgent, AgentConfig
from nova.core.tools import ToolRegistry
from nova.tools.browser import get_browser_tools

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_agent(task: str, headless: bool = True) -> Optional[str]:
    """
    Initialize and run the Nova agent with the given task.
    
    Args:
        task: The task description to execute
        headless: Whether to run the browser in headless mode
        
    Returns:
        The final result of the task execution, or None if an error occurred
    """
    try:
        # Initialize browser
        browser = Browser(headless=headless)
        await browser.start()
        
        # Initialize tool registry
        registry = ToolRegistry()
        
        # Register browser tools
        browser_tools = get_browser_tools(browser)
        for tool in browser_tools:
            registry.register_tool(tool)
            logger.info(f"Registered browser tool: {tool.name}")
        
        # Configure LLM with NVIDIA as primary provider
        llm_config = LLMConfig(
            primary_provider="nvidia",
            primary_model="llama-3.3-70b-instruct",
            primary_base_url=os.getenv("NVIDIA_API_BASE_URL", "http://localhost:8000"),
            primary_api_key=os.getenv("NVIDIA_API_KEY"),
            
            fallback_providers={
                "openai": {
                    "model_name": "gpt-4-turbo-preview",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "base_url": os.getenv("OPENAI_BASE_URL")
                },
                "google": {
                    "model_name": "gemini-pro",
                    "api_key": os.getenv("GOOGLE_API_KEY"),
                    "base_url": os.getenv("GOOGLE_BASE_URL")
                }
            },
            
            # Common parameters
            temperature=0.1,
            max_tokens=4096,
            streaming=True,
            timeout=15
        )
        
        # Configure agent
        agent_config = AgentConfig(
            max_iterations=10,
            max_retries=3,
            tools=registry.get_tools()
        )
        
        # Initialize task agent
        agent = TaskAgent(
            config=agent_config,
            llm_config=llm_config,
            browser=browser
        )
        
        # Run the agent
        result = await agent.run(task)
        
        return result
        
    except Exception as e:
        logger.error(f"Error running agent: {e}", exc_info=True)
        return None
        
    finally:
        # Clean up browser
        if browser:
            await browser.stop()

def main():
    """Main entry point for the CLI."""
    # Initialize components
    registry = ToolRegistry()
    browser = Browser()
    
    # Register browser tools
    browser_tools = get_browser_tools(browser)
    for tool in browser_tools:
        registry.register_tool(tool)
        logger.info(f"Registered browser tool: {tool.name}")

    parser = argparse.ArgumentParser(description="Nova - AI-powered Autonomous Browser Agent")
    
    parser.add_argument(
        "task",
        help="The task description to execute (e.g., 'Search for the latest news about AI')"
    )
    
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser in headless mode (default: True)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--provider",
        choices=["nvidia", "openai", "google"],
        default="nvidia",
        help="Specify the primary LLM provider to use"
    )
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run the agent
    result = asyncio.run(run_agent(args.task, args.headless))
    
    # Print the result
    if result:
        print("\nTask completed successfully!")
        print(f"Result: {result}")
    else:
        print("\nTask failed. Check the logs for details.")

if __name__ == "__main__":
    main() 