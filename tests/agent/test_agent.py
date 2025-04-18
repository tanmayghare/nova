"""Tests for the TaskAgent implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Import necessary components
from nova.core.llm import LLM, LLMConfig
from nova.core.memory import Memory, MemoryConfig
from nova.core.tools import Tool, ToolRegistry, ToolResult, BrowserTools
from nova.core.browser import Browser, BrowserConfig
from nova.agents.task.task_agent import TaskAgent, TaskResult

# --- Fixtures ---

@pytest.fixture
def llm_config():
    """Basic LLMConfig fixture."""
    # Minimal config, assume env vars are mocked elsewhere if needed
    return LLMConfig()

@pytest.fixture
def browser_config():
    """Basic BrowserConfig fixture."""
    return BrowserConfig(headless=True)

@pytest.fixture
def memory_config():
    """Basic MemoryConfig fixture."""
    return MemoryConfig()

# --- Mock Classes/Instances --- 

# Use AsyncMock for async methods
@pytest.fixture
def mock_llm():
    """Mock LLM instance."""
    mock = AsyncMock(spec=LLM)
    # Mock the methods TaskAgent interacts with
    mock.generate_structured_output.return_value = { # Simulate LLM providing next action
        "thought": "I need to navigate first.",
        "action": {
            "tool": "navigate", 
            "input": {"url": "https://example.com"}
        }
    }
    mock.generate_text.return_value = "Task appears complete." # Simulate completion check response
    return mock

@pytest.fixture
def mock_browser():
    """Mock Browser instance."""
    mock = AsyncMock(spec=Browser)
    mock.start = AsyncMock()
    mock.stop = AsyncMock()
    mock.get_current_url = AsyncMock(return_value="mock://blank")
    # BrowserTools might call other methods if used, mock them as needed
    return mock

@pytest.fixture
def mock_memory():
    """Mock Memory instance."""
    mock = AsyncMock(spec=Memory)
    mock.load_memory_variables = AsyncMock(return_value={"history": "mock history"})
    mock.save_context = AsyncMock()
    mock.get_context = AsyncMock(return_value="mock context")
    return mock

@pytest.fixture
def mock_tool_registry():
    """Mock ToolRegistry instance."""
    mock = MagicMock(spec=ToolRegistry) # Use MagicMock for easier attribute/method mocking
    mock.execute_tool = AsyncMock(return_value=ToolResult(success=True, data={"status": "navigated"}).to_dict())
    mock.get_tool_names = MagicMock(return_value=["navigate", "mock_tool"])
    mock.get_tool_schema_string = MagicMock(return_value="Schema: {...}")
    mock.get_tool_by_type = MagicMock(return_value=None) # Default: BrowserTools not found
    mock.register_tool = MagicMock()
    return mock

# --- Tests --- 

@patch('nova.agents.task.task_agent.LLM')
@patch('nova.agents.task.task_agent.Browser')
@patch('nova.agents.task.task_agent.Memory')
@patch('nova.agents.task.task_agent.ToolRegistry')
def test_task_agent_initialization_no_browser(
    MockToolRegistry, MockMemory, MockBrowser, MockLLM,
    llm_config, memory_config # Use fixtures
):
    """Test agent initialization without browser config."""
    mock_llm_instance = MockLLM.return_value
    mock_memory_instance = MockMemory.return_value
    mock_registry_instance = MockToolRegistry.return_value
    mock_browser_instance = MockBrowser.return_value

    class CustomTool(Tool):
        def __init__(self): super().__init__("custom", "desc")
        async def execute(self, **kwargs): return ToolResult(True, {}) 
            
    custom_tools = [CustomTool()]

    agent = TaskAgent(
        task_id="test-01",
        task_description="Do something without browser",
        llm_config=llm_config,
        browser_config=None, # Explicitly None
        memory=mock_memory_instance, # Pass mock instance
        tools=custom_tools 
    )

    MockLLM.assert_called_once_with(config=llm_config)
    assert agent.llm == mock_llm_instance
    assert agent.llm_config == llm_config
    
    assert agent.browser_config is None
    MockBrowser.assert_not_called() # Browser should not be initialized
    assert agent.browser is None

    assert agent.memory == mock_memory_instance
    MockMemory.assert_called_once() # Should be called if memory=None, but we passed instance

    assert agent.tool_registry == mock_registry_instance
    MockToolRegistry.assert_called_once()
    # Check that custom tool was registered
    mock_registry_instance.register_tool.assert_called_once_with(custom_tools[0])
    # Check BrowserTools were NOT registered
    assert mock_registry_instance.get_tool_by_type(BrowserTools) is None


@patch('nova.agents.task.task_agent.LLM')
@patch('nova.agents.task.task_agent.Browser')
@patch('nova.agents.task.task_agent.Memory')
@patch('nova.agents.task.task_agent.ToolRegistry')
@patch('nova.agents.task.task_agent.BrowserTools') # Also patch BrowserTools class
def test_task_agent_initialization_with_browser(
    MockBrowserTools, MockToolRegistry, MockMemory, MockBrowser, MockLLM,
    llm_config, browser_config, memory_config
):
    """Test agent initialization *with* browser config, ensuring BrowserTools are added."""
    mock_llm_instance = MockLLM.return_value
    mock_memory_instance = MockMemory.return_value
    mock_registry_instance = MockToolRegistry.return_value
    mock_browser_instance = MockBrowser.return_value
    mock_browser_tools_instance = MockBrowserTools.return_value

    # Simulate ToolRegistry initially not having BrowserTools
    mock_registry_instance.get_tool_by_type.return_value = None 

    agent = TaskAgent(
        task_id="test-02",
        task_description="Do something with browser",
        llm_config=llm_config,
        browser_config=browser_config, # Provide browser config
        memory=mock_memory_instance,
        tools=[] # No other tools
    )

    assert agent.llm_config == llm_config
    assert agent.browser_config == browser_config
    MockBrowser.assert_called_once_with(config=browser_config)
    assert agent.browser == mock_browser_instance
    
    # Check BrowserTools initialization and registration
    MockBrowserTools.assert_called_once_with(mock_browser_instance)
    mock_registry_instance.register_tool.assert_called_once_with(mock_browser_tools_instance)


# Patch the internal methods for run() test
@patch.object(TaskAgent, '_get_next_action', new_callable=AsyncMock)
@patch.object(TaskAgent, '_execute_action', new_callable=AsyncMock)
@patch.object(TaskAgent, '_is_task_complete', new_callable=AsyncMock)
@patch.object(TaskAgent, 'start', new_callable=AsyncMock)
@patch.object(TaskAgent, 'stop', new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_task_execution_success(
    mock_stop, mock_start, mock_is_task_complete, mock_execute_action, mock_get_next_action,
    llm_config, browser_config, mock_memory # Need memory for save_context
):
    """Test the run loop logic for a successful task execution."""
    # --- Setup Mocks for run() --- 
    # 1. _get_next_action returns an action, then None (or _is_task_complete returns True)
    action1 = {"tool": "navigate", "input": {"url": "https://example.com"}}
    mock_get_next_action.side_effect = [action1, None] # Simulate one action, then stop
    
    # 2. _execute_action returns a successful ToolResult dict
    action1_result = ToolResult(success=True, data={"status": "navigated"}).to_dict()
    mock_execute_action.return_value = action1_result
    
    # 3. _is_task_complete returns False, then True (can also be controlled by _get_next_action)
    # We use side_effect on _get_next_action to stop loop, so this mock isn't strictly needed here
    mock_is_task_complete.return_value = False 

    # --- Initialize Agent (mocks patched externally) --- 
    agent = TaskAgent(
        task_id="run-test-01",
        task_description="Test run success",
        llm_config=llm_config,
        browser_config=browser_config, # Needs browser for start/stop calls
        memory=mock_memory # Use mocked memory
    )
    agent.memory = mock_memory # Ensure mock memory is used
    # agent.tool_registry = mock_tool_registry # Can assign if needed
    # agent.llm = mock_llm # Can assign if needed

    # --- Execute Run --- 
    result = await agent.run()

    # --- Assertions --- 
    mock_start.assert_awaited_once()
    assert mock_get_next_action.await_count == 2 # Called until None is returned
    mock_execute_action.assert_awaited_once_with(action1)
    # mock_is_task_complete.assert_awaited_once_with(action1_result)
    mock_memory.save_context.assert_awaited_once()
    mock_stop.assert_awaited_once()

    assert isinstance(result, TaskResult)
    assert result.status == "completed" # Should infer completion if loop ends gracefully
    assert result.steps_taken == 1 # Executed one action step
    assert result.error is None
    # assert result.result # Check final result content if applicable


@patch.object(TaskAgent, '_get_next_action', new_callable=AsyncMock)
@patch.object(TaskAgent, '_execute_action', new_callable=AsyncMock)
@patch.object(TaskAgent, 'start', new_callable=AsyncMock)
@patch.object(TaskAgent, 'stop', new_callable=AsyncMock)
@patch.object(TaskAgent, '_handle_error') # Mock error handler
@pytest.mark.asyncio
async def test_task_execution_action_error(
    mock_handle_error, mock_stop, mock_start, mock_execute_action, mock_get_next_action,
    llm_config, mock_memory
):
    """Test the run loop logic when _execute_action raises an error."""
    # --- Setup Mocks --- 
    action1 = {"tool": "bad_tool", "input": {}}
    mock_get_next_action.return_value = action1 # Always return action
    
    # _execute_action raises exception
    execution_exception = ValueError("Tool failed miserably")
    mock_execute_action.side_effect = execution_exception

    # --- Initialize Agent --- 
    agent = TaskAgent(
        task_id="run-fail-01",
        task_description="Test run failure",
        llm_config=llm_config,
        browser_config=None, # No browser needed for this test
        memory=mock_memory
    )
    agent.memory = mock_memory

    # --- Execute Run --- 
    result = await agent.run()

    # --- Assertions --- 
    mock_start.assert_awaited_once()
    mock_get_next_action.assert_awaited_once() # Gets action once
    mock_execute_action.assert_awaited_once_with(action1) # Executes once
    mock_handle_error.assert_called_once_with(
        "unexpected_execution_error", 
        {"step": 0, "error": str(execution_exception)}
    )
    mock_memory.save_context.assert_not_called() # Should not save on error before save
    mock_stop.assert_awaited_once()

    assert isinstance(result, TaskResult)
    assert result.status == "failed" # Should fail if error handler called
    assert result.steps_taken == 0 # Error occurred in step 0
    assert str(execution_exception) in result.error # Error should be propagated
