import os
import pytest
import json
from unittest.mock import AsyncMock, MagicMock

from nova.agents.task.task_agent import TaskAgent
from nova.core.config import LLMConfig, AgentConfig
from nova.core.llm import LLM
from nova.core.memory import Memory
from nova.core.browser import Browser
from nova.tools.browser import get_browser_tools

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
    """Provides a TaskAgent instance configured for testing with mocks."""
    # Mock LLM configuration
    mock_llm_config = LLMConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=1000,
        api_key="test_key"
    )
    mock_agent_config = AgentConfig(
        max_iterations=3,
        llm_config=mock_llm_config
    )

    # Mock the LLM instance
    mock_llm = MagicMock(spec=LLM)
    mock_llm.generate_plan.return_value = {
        "steps": [
            {
                "type": "browser",
                "action": {"type": "navigate", "url": "https://example.com"}
            },
            {
                "type": "browser",
                "action": {"type": "click", "selector": "#submit"}
            },
            {
                "type": "finish",
                "action": {"result": "Task completed"}
            }
        ]
    }

    # Mock Browser
    mock_browser = AsyncMock(spec=Browser)
    mock_browser.page.title = AsyncMock(return_value="Test Page")
    mock_browser.page.url = "http://test.com"

    # Mock Memory
    mock_memory = MagicMock(spec=Memory)

    # Create TaskAgent instance
    agent = TaskAgent(
        llm_config=mock_llm_config,
        memory=mock_memory,
        tools=get_browser_tools(mock_browser)
    )

    # Set the browser
    agent.browser = mock_browser

    # Override the interaction_logger path
    if agent.interaction_logger:
        agent.interaction_logger.log_file_path = test_log_file_path
    else:
        pytest.fail("Agent did not initialize interaction_logger successfully")

    # Mock start/stop methods
    agent.start = AsyncMock()
    agent.stop = AsyncMock()
    agent.cleanup = AsyncMock()

    return agent

@pytest.mark.asyncio
async def test_agent_logs_interactions_correctly(test_agent, test_log_file_path):
    """Test that agent logs start, steps, and outcome correctly during a run."""
    test_goal = "Log in to the test website."

    try:
        await test_agent.run(task=test_goal, task_id="test-log-run-001")
    except Exception as e:
        pytest.fail(f"Agent run failed with exception: {e}")

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

    # Check State
    expected_state = "State: Title='Test Page', URL='http://test.com'"
    assert entries[0]["simplified_state"] == expected_state

    # Check Actions
    assert json.loads(entries[0]["action_taken"])["action"]["type"] == "navigate"
    assert json.loads(entries[1]["action_taken"])["action"]["type"] == "click"
    assert json.loads(entries[2]["action_taken"])["action"]["type"] == "finish"

    # Check Final Outcome
    assert all(e["outcome"] == "SUCCESS" for e in entries) 