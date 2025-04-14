import pytest
import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock # For mocking async functions and objects

from nova.agent.agent import Agent # The agent implementation to test
from nova.core.config import AgentConfig, LLMConfig, NIMConfig # For agent configuration
from nova.core.llm import LLM # May need mocking

# --- Integration Tests for Agent Logging ---

@pytest.fixture
def temp_log_dir(tmp_path):
    """Creates a temporary directory specifically for logs in this test."""
    log_dir = tmp_path / "agent_logs"
    log_dir.mkdir()
    return log_dir

@pytest.fixture
def test_log_file_path(temp_log_dir):
    """Path to the log file within the temporary log directory."""
    return str(temp_log_dir / "test_agent_interaction_log.jsonl")

@pytest.fixture
async def test_agent(mocker, test_log_file_path):
    """Provides an Agent instance configured for testing with mocks."""
    # Mock LLM configuration
    mock_llm_config = LLMConfig(
        provider="mock", # Using mock provider
        model_name="mock-model",
        nim_config=NIMConfig(api_base="mock-url", docker_image="mock-image"),
        temperature=0.1,
        max_tokens=100,
        batch_size=1,
        enable_streaming=False
    )
    mock_agent_config = AgentConfig(llm_config=mock_llm_config, max_iterations=3)

    # Mock the LLM instance used by the agent
    # Patch the LLM class directly if Agent creates it internally
    # Or pass a mocked instance if Agent accepts one
    mock_llm_instance = MagicMock(spec=LLM)
    # --- Mock the generate_action method --- 
    # This needs to simulate the ReAct cycle
    actions_to_return = [
        # Iteration 1: Click a button
        {
            "status": "success", 
            "thought": "I need to click the login button.",
            "action": {"tool": "ClickTool", "input": {"selector": "#login-button"}, "type": "tool", "parameters": {}} # Added dummy params
        },
        # Iteration 2: Type username
        {
            "status": "success",
            "thought": "Now I need to type the username.",
            "action": {"tool": "TypeTool", "input": {"selector": "#username", "text": "testuser"}, "type": "tool", "parameters": {}}
        },
        # Iteration 3: Finish
        {
            "status": "success",
            "thought": "Task seems complete.",
            "action": {"tool": "finish", "input": {"result": "Logged in successfully"}, "type": "finish", "parameters": {}}
        },
    ]
    mock_llm_instance.generate_action = AsyncMock(side_effect=actions_to_return)
    # ----------------------------------------

    # Mock Tool Registry and Tool Execution
    mock_tool_registry = MagicMock()
    mock_tool_registry.get_tool_descriptions = MagicMock(return_value="Mock tool descriptions")
    # Simulate successful tool execution - RETURN AN OBJECT WITH .data
    mock_tool_result = MagicMock()
    mock_tool_result.data = "Tool executed successfully"
    mock_tool_result.error = None # Assume no error for successful mock
    mock_tool_registry.execute_tool = AsyncMock(return_value=mock_tool_result)

    # Patch LLM instantiation within Agent.__init__ if necessary
    # If Agent creates LLM internally: mocker.patch('nova.agent.agent.LLM', return_value=mock_llm_instance)
    # If CoreAgent creates LLM internally: mocker.patch('nova.core.agent.LLM', return_value=mock_llm_instance)
    # Assuming CoreAgent creates it based on previous analysis:
    mocker.patch('nova.core.agent.LLM', return_value=mock_llm_instance)

    # Instantiate the agent
    # We might need to mock the Browser and BrowserPool too
    mocker.patch('nova.core.agent.BrowserPool') # Prevent actual browser pool start
    mock_browser = AsyncMock()
    mock_browser.page.title = AsyncMock(return_value="Test Page")
    mock_browser.page.url = "http://test.com"
    mock_browser._page.title = AsyncMock(return_value="Test Page") # Mocking _page as well seems needed sometimes
    mock_browser._page.url = "http://test.com"


    agent = Agent(llm=mock_llm_instance, config=mock_agent_config)
    agent.tool_registry = mock_tool_registry # Inject mock tool registry
    agent.browser = mock_browser # Inject mock browser
    
    # --- Crucially, override the interaction_logger path --- 
    # Ensure the logger instance exists before trying to set its path
    if agent.interaction_logger:
         agent.interaction_logger.log_file_path = test_log_file_path
    else:
         pytest.fail("Agent did not initialize interaction_logger successfully")
    # ------------------------------------------------------

    # Mock the start/stop methods to avoid external dependencies
    agent.start = AsyncMock()
    agent.stop = AsyncMock()
    agent.cleanup = AsyncMock()

    # Mock _get_structured_dom if it's called within the loop
    # agent._get_structured_dom = AsyncMock(return_value=[{"tag": "div", "text": "mock dom element"}])
    # Let's assume the base implementation uses page title/URL for logging state now

    return agent

@pytest.mark.asyncio # Mark test as async
async def test_agent_logs_interactions_correctly(test_agent, test_log_file_path):
    """Test that agent logs start, steps, and outcome correctly during a run."""
    test_goal = "Log in to the test website."

    # --- Run the agent task ---
    # The run method now encapsulates the _execute_task loop
    try:
        await test_agent.run(task=test_goal, task_id="test-log-run-001")
    except Exception as e:
        # Allow seeing the exception if the run fails unexpectedly
        pytest.fail(f"Agent run failed with exception: {e}")

    # --- Assert Log File Content --- 
    assert os.path.exists(test_log_file_path), "Log file was not created"

    with open(test_log_file_path, 'r') as f:
        log_lines = f.readlines()
    
    assert len(log_lines) == 3, f"Expected 3 log entries, found {len(log_lines)}"

    entries = [json.loads(line) for line in log_lines]

    # Check Session ID is consistent
    session_id = entries[0]["session_id"]
    assert all(e["session_id"] == session_id for e in entries)

    # Check Step IDs
    assert entries[0]["step_id"] == 0
    assert entries[1]["step_id"] == 1
    assert entries[2]["step_id"] == 2

    # Check User Goal
    assert all(e["user_goal"] == test_goal for e in entries)

    # Check Simplified State (using mock page title/URL fallback for first step)
    # First step uses fallback because dom_context_str is initial
    expected_state_step1 = "State: Title='Test Page', URL='http://test.com'"
    # Subsequent steps use the result of _get_structured_dom (which is mocked to return [])
    expected_state_subsequent = "[]"
    assert entries[0]["simplified_state"] == expected_state_step1
    assert entries[1]["simplified_state"] == expected_state_subsequent
    assert entries[2]["simplified_state"] == expected_state_subsequent

    # Check Actions Taken (based on mocked LLM responses)
    assert json.loads(entries[0]["action_taken"])["tool"] == "ClickTool"
    assert json.loads(entries[1]["action_taken"])["tool"] == "TypeTool"
    assert json.loads(entries[2]["action_taken"])["tool"] == "finish"

    # Check Final Outcome (should be SUCCESS as mocked LLM returned 'finish')
    assert all(e["outcome"] == "SUCCESS" for e in entries) 