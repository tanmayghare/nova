import pytest
from typing import Any, Dict, Sequence
from unittest.mock import AsyncMock, MagicMock
from pytest_mock import MockerFixture

from nova.agent.agent import Agent
from nova.core.browser import Browser
from nova.core.config import AgentConfig, BrowserConfig
from nova.core.llm import LLM
from nova.core.memory import Memory
from nova.core.tools import Tool


class MockTool(Tool):
    """A mock tool for testing."""

    def __init__(self) -> None:
        super().__init__("mock", "A mock tool")

    async def execute(self, input_data: Dict[str, Any]) -> str:
        return f"Mock result for {input_data}"


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM."""
    llm = MagicMock(spec=LLM)
    # Updated for ReAct: generate_plan returns (thought, plan_steps)
    # Use side_effect to simulate loop: 1 action, then stop
    llm.generate_plan = AsyncMock(side_effect=[
        ("Thought for step 1", [{'tool': 'mock', 'input': {'test': 'data'}}]), # Iteration 1
        ("Thought for stopping", []) # Iteration 2 (empty plan stops loop)
    ])
    llm.generate_response = AsyncMock(return_value="Final ReAct response")
    return llm


@pytest.fixture
def mock_browser() -> MagicMock:
    """Create a mock browser."""
    browser = MagicMock(spec=Browser)
    browser.start = AsyncMock()
    browser.stop = AsyncMock()
    browser.navigate = AsyncMock()
    browser.get_text = AsyncMock(return_value="Example text")
    
    # Add mock page attribute for _get_structured_dom
    browser.page = AsyncMock()
    # Ensure screenshot returns a serializable path string
    browser.screenshot = AsyncMock(return_value="screenshots/mock_screenshot.png") 

    return browser


@pytest.fixture
def mock_memory() -> MagicMock:
    """Create a mock memory."""
    memory = MagicMock(spec=Memory)
    memory.get_context = AsyncMock(return_value="Context")
    memory.add = AsyncMock()
    return memory


@pytest.fixture
async def agent(mock_llm: MagicMock, mock_browser: MagicMock, mock_memory: MagicMock) -> Agent:
    """Create an agent instance for testing."""
    tools: Sequence[Tool] = [MockTool()]
    config = AgentConfig()
    browser_config = BrowserConfig(headless=True)
    agent_instance = Agent(
        llm=mock_llm,
        tools=tools,
        memory=mock_memory,
        config=config,
        browser_config=browser_config,
    )
    # Manually assign the mock browser AFTER initialization for testing purposes,
    # as the actual browser initialization happens within CoreAgent's __init__
    # which we don't want to interfere with heavily.
    agent_instance.browser = mock_browser 
    
    yield agent_instance
    
    # Cleanup
    if agent_instance.browser:
        await agent_instance.browser.stop()


@pytest.mark.asyncio
async def test_agent_initialization(
    agent: Agent,
    mock_llm: MagicMock,
    mock_browser: MagicMock,
    mock_memory: MagicMock,
) -> None:
    """Test agent initialization."""
    assert agent.llm == mock_llm
    assert len(agent.tools) == 1
    assert agent.memory == mock_memory
    assert isinstance(agent.config, AgentConfig)
    assert isinstance(agent.browser, Browser)


@pytest.mark.asyncio
async def test_agent_run(agent: Agent, mock_llm: MagicMock, mock_browser: MagicMock, mock_memory: MagicMock, mocker: MockerFixture) -> None:
    """Test running the agent with the ReAct loop."""
    task_description = "Test ReAct task"
    
    # Mock execute_tool using mocker for this specific test
    mock_execute = mocker.patch.object(agent.tool_registry, 'execute_tool', return_value="Mock tool execution result")

    result = await agent.run(task_description)

    # Check final status and response
    assert result["status"] == "success"
    assert result["response"] == "Final ReAct response"
    assert result["results"] == ["Mock tool execution result"] # Should contain result from the one executed tool

    # Check history
    assert isinstance(result["history"], list)
    assert len(result["history"]) == 2 # 1 iteration with action, 1 iteration stopping
    
    # Check first iteration history entry
    history_1 = result["history"][0]
    assert history_1["iteration"] == 1
    assert history_1["thought"] == "Thought for step 1"
    assert history_1["action"] == {'tool': 'mock', 'input': {'test': 'data'}}
    assert history_1["observation"]["status"] == "success"
    assert history_1["observation"]["result"] == "Mock tool execution result"

    # Check second iteration history entry (stopping)
    history_2 = result["history"][1]
    assert history_2["iteration"] == 2
    assert history_2["thought"] == "Thought for stopping"
    assert "action" not in history_2 # No action taken
    assert "observation" not in history_2 # No observation

    # Verify LLM calls
    assert mock_llm.generate_plan.call_count == 2 # Called twice (once for action, once to stop)
    # Check arguments for the first call (more detailed check)
    first_call_args, _ = mock_llm.generate_plan.call_args_list[0]
    assert first_call_args[0] == task_description # task
    
    # --- Update Assertion for Context --- 
    # Expect the fully formatted context, even on the first call
    # mock_get_dom returns [] in this test by default from the mock page
    initial_context_val = "Context" # From mock_memory
    expected_first_context = f"""
Initial Task Context:
{initial_context_val}

Current Page Structure (Interactive Elements):
```json
[]
```

Recent Execution History (last 0 steps):
```json
[]
```
"""
    # Compare stripped versions to ignore potential leading/trailing whitespace differences
    assert first_call_args[1].strip() == expected_first_context.strip()
    # --- End Update --- 
    
    # Check arguments for the second call (context should include history)
    second_call_args, _ = mock_llm.generate_plan.call_args_list[1]
    assert second_call_args[0] == task_description
    assert "Recent Execution History" in second_call_args[1]
    assert "Thought for step 1" in second_call_args[1]
    assert "Mock tool execution result" in second_call_args[1]
    assert "screenshots/mock_screenshot.png" in second_call_args[1]
    
    mock_llm.generate_response.assert_called_once()
    response_call_args, _ = mock_llm.generate_response.call_args_list[0]
    assert response_call_args[0] == task_description
    assert response_call_args[1] == result["history"] # History passed to generate final response
    assert "Task completed or max iterations reached" in response_call_args[2]
    
    # Verify memory calls
    mock_memory.get_context.assert_called() # Called multiple times (initial + context updates)
    mock_memory.add.assert_called_once()
    add_call_args, _ = mock_memory.add.call_args
    assert add_call_args[1] == {'tool': 'mock', 'input': {'test': 'data'}} # Step added
    assert add_call_args[2]["status"] == "success" # Observation added
    
    # Verify browser lifecycle
    mock_browser.start.assert_called_once()
    mock_browser.stop.assert_called_once()
    
    # Verify tool execution call (using the mocker-patched object)
    mock_execute.assert_called_once_with('mock', {'test': 'data'})


@pytest.mark.asyncio
async def test_agent_run_max_failures(agent: Agent, mock_llm: MagicMock, mock_memory: MagicMock, mocker: MockerFixture) -> None:
    """Test that the agent run stops after reaching max consecutive tool failures."""
    task_description = "Test max failures task"
    max_fails = 2
    agent.config.max_failures = max_fails
    
    mock_llm.generate_plan.side_effect = None 
    mock_llm.generate_plan.return_value = ("Thinking about failing", [{'tool': 'mock', 'input': {'fail': 'me'}}])
    
    # Mock Tool Registry to always fail using mocker
    tool_error = RuntimeError("Tool execution failed consistently")
    mock_execute = mocker.patch.object(agent.tool_registry, 'execute_tool', side_effect=tool_error)
    
    result = await agent.run(task_description)
    
    # Assertions
    assert result["status"] == "failed"
    assert "Reached max tool failures" in result["error"] 
    assert str(tool_error) in result["error"] # Check if original error is included
    assert len(result["history"]) == max_fails # Should have tried max_fails times
    
    # Verify calls (using the mocker-patched object)
    assert mock_execute.call_count == max_fails
    mock_llm.generate_response.assert_not_called() # Should fail before generating final response
    mock_memory.add.assert_called() # Memory should have been called to record errors
    assert mock_memory.add.call_count == max_fails
    # Check that the last add call recorded the error observation
    last_add_call_args, _ = mock_memory.add.call_args_list[-1]
    assert last_add_call_args[2]["status"] == "error"
    assert last_add_call_args[2]["error"] == str(tool_error)


@pytest.mark.asyncio
async def test_agent_run_finish_tool(agent: Agent, mock_llm: MagicMock, mock_memory: MagicMock, mocker: MockerFixture) -> None:
    """Test that the agent run stops correctly when the LLM returns the finish tool."""
    task_description = "Test finish tool task"
    finish_reason = "Task completed successfully"
    tool_result = "Result from mock tool"

    mock_llm.generate_plan.side_effect = [
        ("Thought for action", [{'tool': 'mock', 'input': {'data': 'step1'}}]),
        ("Thought for finishing", [{'tool': 'finish', 'input': {'reason': finish_reason}}])
    ]
    mock_llm.generate_response.return_value = "Final response after finish"
    
    # Mock Tool Registry to succeed on the first call using mocker
    mock_execute = mocker.patch.object(agent.tool_registry, 'execute_tool', return_value=tool_result)
    
    result = await agent.run(task_description)
    
    # Assertions
    assert result["status"] == "success"
    assert result["response"] == "Final response after finish"
    assert result["results"] == [tool_result] # Only the result from the first action
    assert len(result["history"]) == 2

    # Check history entries
    history_1 = result["history"][0]
    assert history_1["thought"] == "Thought for action"
    assert history_1["action"] == {'tool': 'mock', 'input': {'data': 'step1'}}
    assert history_1["observation"]["status"] == "success"
    assert history_1["observation"]["result"] == tool_result

    history_2 = result["history"][1]
    assert history_2["thought"] == "Thought for finishing"
    assert history_2["action"]["tool"] == "finish"
    assert history_2["action"]["input"]["reason"] == finish_reason
    assert history_2["observation"]["status"] == "success"
    assert "Task marked as finished by LLM" in history_2["observation"]["result"]
    
    # Verify calls (using the mocker-patched object)
    assert mock_execute.call_count == 1
    mock_execute.assert_called_once_with('mock', {'data': 'step1'})
    assert mock_llm.generate_plan.call_count == 2
    mock_llm.generate_response.assert_called_once()
    mock_memory.add.assert_called_once() # Only called for the successful 'mock' tool execution
