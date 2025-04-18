"""Tests for tool functionality."""

import pytest
import time
from unittest.mock import AsyncMock, patch

from nova.core.tools import Tool, ToolRegistry, ToolResult
from nova.tools.browser import (
    NavigateTool,
    ClickTool,
    TypeTool,
    GetTextTool,
    GetHtmlTool,
    ScreenshotTool,
    WaitTool,
    ScrollTool
)

class MockTool(Tool):
    """Mock tool for testing."""
    
    def __init__(self):
        super().__init__("mock", "A mock tool")
        self.execute = AsyncMock(return_value={"status": "success", "result": "Mock result"})

@pytest.fixture
def tool_registry():
    return ToolRegistry()

@pytest.fixture
def mock_tool():
    return MockTool()

@pytest.mark.asyncio
async def test_tool_registration(tool_registry, mock_tool):
    """Test tool registration."""
    # Register tool
    tool_registry.register_tool(mock_tool)
    
    # Verify registration
    assert len(tool_registry.get_tools()) == 1
    assert tool_registry.get_tool("mock") == mock_tool
    
    # Test duplicate registration
    with pytest.raises(ValueError):
        tool_registry.register_tool(mock_tool)

@pytest.mark.asyncio
async def test_tool_execution(mock_tool):
    """Test tool execution."""
    # Execute tool
    result = await mock_tool.execute({"test": "data"})
    
    # Verify execution
    assert result["status"] == "success"
    assert result["result"] == "Mock result"
    mock_tool.execute.assert_called_once_with({"test": "data"})

@pytest.mark.asyncio
async def test_browser_tools():
    """Test browser tool implementations."""
    # Create browser tools
    navigate_tool = NavigateTool()
    click_tool = ClickTool()
    type_tool = TypeTool()
    get_text_tool = GetTextTool()
    get_html_tool = GetHtmlTool()
    screenshot_tool = ScreenshotTool()
    wait_tool = WaitTool()
    scroll_tool = ScrollTool()
    
    # Test tool names
    assert navigate_tool.name == "navigate"
    assert click_tool.name == "click"
    assert type_tool.name == "type"
    assert get_text_tool.name == "get_text"
    assert get_html_tool.name == "get_html"
    assert screenshot_tool.name == "screenshot"
    assert wait_tool.name == "wait"
    assert scroll_tool.name == "scroll"
    
    # Test tool descriptions
    assert "navigate" in navigate_tool.description.lower()
    assert "click" in click_tool.description.lower()
    assert "type" in type_tool.description.lower()
    assert "text" in get_text_tool.description.lower()
    assert "html" in get_html_tool.description.lower()
    assert "screenshot" in screenshot_tool.description.lower()
    assert "wait" in wait_tool.description.lower()
    assert "scroll" in scroll_tool.description.lower()

@pytest.mark.asyncio
async def test_tool_error_handling(mock_tool):
    """Test tool error handling."""
    # Test execution error
    mock_tool.execute.side_effect = Exception("Tool execution failed")
    with pytest.raises(Exception) as exc_info:
        await mock_tool.execute({"test": "data"})
    assert str(exc_info.value) == "Tool execution failed"
    
    # Test invalid input
    with pytest.raises(ValueError):
        await mock_tool.execute(None)
    
    # Test missing required parameters
    with pytest.raises(ValueError):
        await mock_tool.execute({}) 

# --- New Tests for ToolResult and execute_tool handling --- 

def test_tool_result_to_dict():
    """Test the to_dict method of ToolResult."""
    start_time = time.time()
    result = ToolResult(
        success=True,
        data={"key": "value"},
        error=None,
        execution_time=time.time() - start_time,
        metadata={"tool_name": "test_tool"}
    )
    result_dict = result.to_dict()

    assert isinstance(result_dict, dict)
    assert result_dict["success"] is True
    assert result_dict["data"] == {"key": "value"}
    assert result_dict["error"] is None
    assert isinstance(result_dict["execution_time"], float)
    assert result_dict["execution_time"] >= 0
    assert result_dict["metadata"] == {"tool_name": "test_tool"}

    # Test with error
    error_result = ToolResult(
        success=False,
        data=None,
        error="Something went wrong",
        execution_time=0.1,
        metadata={}
    )
    error_dict = error_result.to_dict()
    assert error_dict["success"] is False
    assert error_dict["data"] is None
    assert error_dict["error"] == "Something went wrong"

class MockToolReturningToolResult(Tool):
    """Mock tool that returns a ToolResult object."""
    def __init__(self, name="mock_result_tool", description="Returns ToolResult"):
        super().__init__(name, description)
        # Define an output schema if needed for validation testing
        # self.output_schema = Mock(spec=BaseModel)
        # self.output_schema.model_validate.return_value = {"output_key": "validated_value"}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            data={"input_args": kwargs, "processed": True},
            error=None,
            execution_time=0.05,
            metadata={"instance_id": 123}
        )

@pytest.mark.asyncio
@patch('nova.core.tools.tools.ToolRegistry._validate_output') # Patch validation
async def test_registry_execute_tool_with_tool_result(mock_validate_output, tool_registry):
    """Test ToolRegistry.execute_tool correctly handles ToolResult return types."""
    mock_result_tool = MockToolReturningToolResult()
    tool_registry.register_tool(mock_result_tool)

    tool_name = "mock_result_tool"
    tool_input = {"arg1": "value1"}

    # Expected dictionary from ToolResult.to_dict()
    expected_dict = {
        'success': True,
        'data': {'input_args': {'arg1': 'value1'}, 'processed': True},
        'error': None,
        'execution_time': 0.05,
        'metadata': {'instance_id': 123}
    }

    # Configure the mock validation to just return the input it receives
    mock_validate_output.side_effect = lambda result_dict, schema: result_dict

    final_result = await tool_registry.execute_tool(tool_name, tool_input)

    # 1. Check that the validation method was called
    mock_validate_output.assert_called_once()

    # 2. Check that the validation method received the dictionary, not the ToolResult object
    call_args, call_kwargs = mock_validate_output.call_args
    received_dict = call_args[0]
    assert isinstance(received_dict, dict)
    assert received_dict == expected_dict
    # assert call_args[1] == mock_result_tool.output_schema # Check schema if defined

    # 3. Check that the final result is the (potentially validated) dictionary
    assert final_result == expected_dict 