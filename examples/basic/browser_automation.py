"""Example demonstrating basic browser automation with Nova."""

import asyncio
import os
from dotenv import load_dotenv

from nova.core.agent import Agent
from nova.core.config import AgentConfig, BrowserConfig
from nova.core.llm import LLM
from nova.core.memory import Memory
from nova.core.llama import LlamaModel


async def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Llama model
    llama_model = LlamaModel(
        model_path=os.getenv("LLAMA_MODEL_PATH"),
        n_ctx=int(os.getenv("LLAMA_N_CTX", "2048")),
        n_threads=int(os.getenv("LLAMA_N_THREADS", "4")),
    )
    
    # Create LLM wrapper
    llm = LLM(model=llama_model, model_type="llama")
    
    # Create configurations
    config = AgentConfig(
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "10")),
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0.1")),
    )
    
    # Configure browser to run in headed mode for visualization
    browser_config = BrowserConfig(
        headless=False,  # Show the browser window
        timeout=int(os.getenv("BROWSER_TIMEOUT", "30000")),
        viewport={
            "width": os.environ.get("BROWSER_VIEWPORT_WIDTH"),
            "height": os.environ.get("BROWSER_VIEWPORT_HEIGHT"),
        },  # Set a reasonable window size
    )

    # Create agent
    agent = Agent(
        llm=llm,
        memory=Memory(),
        config=config,
        browser_config=browser_config,
    )
    
    # Define a simple task
    task = """
    Navigate to https://example.com
    Get the page title
    Get the text content of the h1 element
    """
    
    print("Executing task...")
    print("Browser window will open. Watch the automation in action!")
    result = await agent.run(task)
    print("\nTask Result:")
    print(result)


if __name__ == "__main__":
    asyncio.run(main()) 