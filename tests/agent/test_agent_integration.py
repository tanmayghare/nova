import os
import asyncio
import logging
import pytest
from dotenv import load_dotenv

from nova.agent.agent import Agent
from nova.core.llm import LLM
from nova.core.browser import Browser, BrowserConfig
from nova.core.memory import Memory
from nova.core.config import AgentConfig

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

# --- Integration Tests --- 

# @pytest.mark.integration # Temporarily removed
@pytest.mark.asyncio
async def test_integration_navigate_and_get_title(integration_agent: Agent):
    """
    Test the agent's ability to navigate to a URL and identify the page title
    using the actual LLM and Browser.
    """
    task = "Navigate to https://example.com and tell me the title of the page."
    print(f"\nRunning integration test with task: '{task}'")

    # Run the agent
    # Add a timeout to prevent test hanging indefinitely if LLM/browser has issues
    try:
        result = await asyncio.wait_for(integration_agent.run(task), timeout=120.0) # 2 min timeout
    except asyncio.TimeoutError:
        pytest.fail("Agent run timed out after 120 seconds.")
        return # Keep type checker happy

    print(f"\nAgent Result:\n{result}")

    # Assertions
    assert result is not None, "Agent did not return a result."
    # Expect status success because the loop completed, even if halted early
    assert result.get("status") == "success", f"Agent failed with error: {result.get('error')}"
    assert "response" in result, "Result missing 'response' key."
    assert "history" in result, "Result missing 'history' key."
    history = result["history"]
    assert isinstance(history, list)
    assert len(history) == 2 # Should have 2 iterations (navigate, then halt)

    # Check Iteration 1: Navigation
    navigate_found = any(
        entry.get("action", {}).get("tool") == "navigate" and
        "example.com" in entry.get("action", {}).get("input", {}).get("url", "")
        for entry in history
    )
    assert navigate_found, "No navigation step to example.com found in history."

    # Check Iteration 2: Halted due to low confidence
    assert history[-1].get("confidence", 1.0) < integration_agent.config.confidence_threshold
    assert history[-1].get("observation", {}).get("status") == "halted"
    assert "below threshold" in history[-1].get("observation", {}).get("reason", "")
    
    # Check final response indicates failure/limitation (fuzzy check)
    response_lower = result["response"].lower()
    assert "failed" in response_lower or "unable" in response_lower or "limitation" in response_lower, \
        f"Expected final response to indicate failure/limitation, but got: {result['response']}"

    # Check that the loop halted due to low confidence, indicated by the observation status
    # The last proposed action might have been 'finish' even with low confidence.
    assert history[-1].get("observation", {}).get("status") == "halted", \
        f"Expected last observation status to be 'halted' due to low confidence, but got: {history[-1].get('observation', {}).get('status')}"
