import pytest
from pathlib import Path
from nova.tools.utils.loader import ToolLoader
from nova.tools.utils.registry import ToolRegistry

def test_tool_loader_initialization():
    """Test that the tool loader initializes correctly."""
    loader = ToolLoader()
    assert isinstance(loader.base_path, Path)
    assert loader.base_path.name == "tools"

def test_discover_tools():
    """Test that the tool loader can discover tool modules."""
    loader = ToolLoader()
    tools = loader.discover_tools()
    assert isinstance(tools, list)
    assert len(tools) > 0
    assert all(isinstance(tool, str) for tool in tools)

def test_load_tool():
    """Test that the tool loader can load a tool module."""
    loader = ToolLoader()
    registry = ToolRegistry()
    
    # Test loading a known tool
    tool = loader.load_tool("calculator", registry)
    assert tool is not None
    assert hasattr(tool, "name")
    assert hasattr(tool, "description")
    assert hasattr(tool, "execute")

def test_load_nonexistent_tool():
    """Test that loading a nonexistent tool raises an appropriate error."""
    loader = ToolLoader()
    registry = ToolRegistry()
    
    with pytest.raises(ImportError):
        loader.load_tool("nonexistent_tool", registry)

def test_tool_registration():
    """Test that loaded tools are properly registered."""
    loader = ToolLoader()
    registry = ToolRegistry()
    
    tool = loader.load_tool("calculator", registry)
    assert tool.name in registry.tools
    assert registry.tools[tool.name] == tool 