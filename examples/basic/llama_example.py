"""Example demonstrating how to use the Llama model with Nova."""

import asyncio
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from nova.core.agent import Agent
from nova.core.config import AgentConfig
from nova.core.llm import LLM
from nova.core.memory import Memory
from nova.core.tools import Tool
from nova.core.llama import LlamaModel


class MockTool(Tool):
    """A mock tool for demonstration."""

    def __init__(self):
        super().__init__(
            name="mock",
            description="A mock tool that returns the input data"
        )

    async def execute(self, input_data: dict) -> str:
        return f"Mock tool executed with input: {input_data}"


async def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize Llama model with configuration from environment variables
    llama_model = LlamaModel(
        model_path=os.getenv("LLAMA_MODEL_PATH"),
        n_ctx=int(os.getenv("LLAMA_N_CTX", "4096")),
        n_threads=int(os.getenv("LLAMA_N_THREADS", "6")),
        n_gpu_layers=int(os.getenv("LLAMA_N_GPU_LAYERS", "1")),
    )
    
    # Create LLM wrapper
    llm = LLM(model=llama_model, model_type="llama")
    
    # Create agent configuration
    config = AgentConfig(
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "10")),
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0.7")),
    )
    
    # Create agent
    agent = Agent(
        llm=llm,
        tools=[MockTool()],
        memory=Memory(),
        config=config,
    )
    
    # Run the agent with a simple task
    task = "Navigate to example.com and get the title"
    print(f"Executing task: {task}")
    result = await agent.run(task)
    print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main()) 