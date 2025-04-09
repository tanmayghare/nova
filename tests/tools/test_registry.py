import pytest
from nova.tools.base.registry import ToolRegistry
from nova.tools.base.tool import BaseTool, ToolConfig, ToolResult
from typing import Dict, Any

class MockTool(BaseTool):
    """A mock tool for testing purposes."""
    
    @classmethod
    def get_default_config(cls) -> ToolConfig:
        return ToolConfig(
            name="mock_tool",
            description="A mock tool for testing"
        )
    
    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, data="mock result")

@pytest.fixture
def registry():
    return ToolRegistry()

@pytest.fixture
def mock_tool():
    return MockTool()

def test_registry_initialization(registry):
    """Test that the registry initializes with empty tool collections."""
    assert registry._tools == {}
    assert registry._tool_classes == {}
    assert registry._tool_configs == {}

def test_register_tool(registry, mock_tool):
    """Test that tools can be registered successfully."""
    registry.register(mock_tool)
    assert mock_tool.config.name in registry._tools
    assert registry._tools[mock_tool.config.name] == mock_tool

def test_register_duplicate_tool(registry, mock_tool):
    """Test that registering a duplicate tool updates the existing registration."""
    registry.register(mock_tool)
    registry.register(mock_tool)  # Should not raise an error
    assert len(registry._tools) == 1
    assert registry._tools[mock_tool.config.name] == mock_tool

def test_get_tool(registry, mock_tool):
    """Test that tools can be retrieved by their name."""
    registry.register(mock_tool)
    retrieved_tool = registry.get_tool(mock_tool.config.name)
    assert retrieved_tool == mock_tool

def test_get_nonexistent_tool(registry):
    """Test that getting a nonexistent tool returns None."""
    assert registry.get_tool("nonexistent") is None

def test_list_tools(registry, mock_tool):
    """Test that the registry can list all registered tools."""
    registry.register(mock_tool)
    tools = registry.list_tools()
    assert isinstance(tools, list)
    assert mock_tool.config.name in tools 