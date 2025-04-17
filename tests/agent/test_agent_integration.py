import os
import asyncio
import logging
import pytest
import json
from dotenv import load_dotenv
from unittest.mock import AsyncMock, patch

from nova.agent.agent import Agent
from nova.core.llm import LLM
from nova.core.browser import Browser, BrowserConfig
from nova.core.memory import Memory
from nova.core.config import AgentConfig
from nova.agents.task.task_agent import TaskAgent, TaskResult
from nova.agents.task.config import TaskAgentConfig
from nova.core.tools import Tool, ToolRegistry
from nova.core.llm import LLMConfig

logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

@pytest.fixture(scope="function")
async def integration_agent():
    """Fixture to create an Agent instance (function scope) for integration testing."""
    # Configure components (adjust based on user's setup)
    # Example: Use environment variables for LLM config if needed
    llm_provider = os.getenv("INTEGRATION_LLM_PROVIDER", "nim") # Default to nim
    llm_model = os.getenv("INTEGRATION_LLM_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1")
    llm_api_base = os.getenv("INTEGRATION_LLM_API_BASE", "http://localhost:8000")
    # Add API key loading if necessary: os.getenv("LLM_API_KEY")

    print(f"\n--- Integration Test Setup ---")
    print(f"Using LLM Provider: {llm_provider}")
    print(f"Using LLM Model: {llm_model}")
    print(f"Using LLM API Base: {llm_api_base}")
    print(f"-----------------------------")


    llm = LLM(
        provider=llm_provider,
        model_name=llm_model,
        api_base=llm_api_base
        # Pass other args like api_key if needed
    )
    # Use default browser config (headless=True might be good for CI)
    # --- Run Headed for Debugging ---
    logger.info("Setting browser to run in headed mode for debugging.")
    browser_config = BrowserConfig(headless=False)
    # --- End Headed Mode --- 
    # Using single Browser instance for simplicity in testing
    # Replace with BrowserPool if testing pooling
    browser = Browser(config=browser_config)
    memory = Memory()
    config = AgentConfig() # Use default agent config for now

    agent_instance = Agent(
        llm=llm,
        tools=[], # Start with no extra tools, agent adds browser tools
        memory=memory,
        config=config,
        browser_config=browser_config # Ensure browser_config is passed if needed for pool fallback
    )
    # Set the browser using the property setter AFTER initialization
    agent_instance.browser = browser 

    try:
        print("Starting agent for integration test...")
        await agent_instance.start()
        print("Agent started.")
        yield agent_instance
    finally:
        # Add a small sleep to allow pending tasks to settle before closing
        await asyncio.sleep(0.1) 
        print("Stopping agent after integration test...")
        await agent_instance.stop()
        print("Agent stopped.")

# --- Fixture for Integration TaskAgent --- 

@pytest.fixture(scope="function") # function scope ensures fresh agent for each test
async def task_agent(request): # request fixture allows passing params
    """Fixture to create and clean up a TaskAgent instance for integration testing."""
    
    task_description = getattr(request, "param", "Navigate to example.com and get the main heading.")
    task_id = f"integration-test-{request.node.name}" # Unique ID based on test name
    
    print(f"\n--- Integration Test: {request.node.name} --- ")
    print(f"Task: {task_description}")

    agent_instance = None
    try:
        # Initialize configs - they will read from .env
        llm_config = LLMConfig()
        # Ensure browser is headed for easier debugging if needed, but default to env var or True
        headless_str = os.getenv("INTEGRATION_BROWSER_HEADLESS", "true").lower()
        headless = headless_str == 'true'
        browser_config = BrowserConfig(headless=headless)
        
        print(f"Using LLM Provider: {llm_config.primary_provider}")
        print(f"Using LLM Model: {llm_config.primary_model}")
        print(f"Browser Headless: {browser_config.headless}")
        print("-----------------------------------")

        # Create the TaskAgent instance
        agent_instance = TaskAgent(
            task_id=task_id,
            task_description=task_description,
            llm_config=llm_config,
            browser_config=browser_config,
            # Use default Memory and ToolRegistry setup within TaskAgent
            memory=None,
            tools=None 
        )
        
        # No need to call start() here, agent.run() handles it.
        yield agent_instance
        
    finally:
        # Cleanup: Stop the agent (which should stop the browser)
        if agent_instance:
            print(f"\n--- Tearing down agent for {request.node.name} ---")
            try:
                await agent_instance.stop()
                print("Agent stopped successfully.")
            except Exception as e:
                logger.error(f"Error stopping agent during teardown: {e}", exc_info=True)
            print("-----------------------------------")

# --- Integration Tests --- 

# Marker to easily run/skip integration tests
@pytest.mark.integration 
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "task_agent", 
    ["Navigate to https://example.com and tell me the main heading (H1 element)."],
    indirect=True # Pass the task description to the fixture
)
async def test_integration_navigate_and_get_heading(task_agent: TaskAgent):
    """
    Test the agent's ability to navigate to example.com and extract the heading
    using the actual LLM and Browser configured via .env.
    """
    print(f"\nStarting test_integration_navigate_and_get_heading...")
    timeout = 180.0 # 3 minutes

    try:
        result: TaskResult = await asyncio.wait_for(task_agent.run(), timeout=timeout)
    except asyncio.TimeoutError:
        pytest.fail(f"Agent run timed out after {timeout} seconds.")
        return # Keep type checker happy

    print(f"\nAgent Result:\n{json.dumps(result.dict(), indent=2)}")

    # Assertions
    assert result is not None, "Agent did not return a result."
    assert result.status == "completed", f"Agent did not complete successfully. Status: {result.status}, Error: {result.error}"
    assert result.error is None, f"Agent finished with an error: {result.error}"
    assert result.steps_taken > 0, "Agent completed in 0 steps, indicating it might not have run properly."
    
    # Check final result data (might vary based on LLM response format)
    # This requires the LLM to actually return the heading in the final step's result.
    # We make this check flexible.
    final_result_str = str(result.result).lower()
    assert "example domain" in final_result_str, \
        f"Expected 'Example Domain' to be mentioned in the final result, but got: {result.result}"

@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "task_agent", 
    ["Navigate to duckduckgo.com, search for the term 'large language model', and report the title of the first result."],
    indirect=True
)
async def test_integration_search_and_get_first_result(task_agent: TaskAgent):
    """
    Test the agent's ability to perform a web search and extract information
    using the actual LLM and Browser configured via .env.
    """
    print(f"\nStarting test_integration_search_and_get_first_result...")
    timeout = 240.0 # 4 minutes (searches can take longer)

    try:
        result: TaskResult = await asyncio.wait_for(task_agent.run(), timeout=timeout)
    except asyncio.TimeoutError:
        pytest.fail(f"Agent run timed out after {timeout} seconds.")
        return

    print(f"\nAgent Result:\n{json.dumps(result.dict(), indent=2)}")

    # Assertions
    assert result is not None, "Agent did not return a result."
    assert result.status == "completed", f"Agent did not complete successfully. Status: {result.status}, Error: {result.error}"
    assert result.error is None, f"Agent finished with an error: {result.error}"
    assert result.steps_taken > 1, "Agent should take multiple steps for searching."

    # Check final result - less strict, just verify it seems plausible
    final_result_str = str(result.result).lower()
    assert len(final_result_str) > 5, "Final result seems too short."
    # Check history for evidence of searching and getting text
    history_str = json.dumps(result.history).lower()
    assert "duckduckgo.com" in history_str, "History does not show navigation to DuckDuckGo."
    assert "search" in history_str or "type" in history_str, "History does not show a search or type action."
    assert "large language model" in history_str, "Search term missing from history."
    assert "get_text" in history_str or "extract" in history_str, "History does not show text extraction."
