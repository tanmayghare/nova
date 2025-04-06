from typing import Any, Dict, List, Optional


class Tool:
    """Base class for tools."""

    def __init__(self, name: str, description: str) -> None:
        """Initialize a tool with a name and description."""
        self.name = name
        self.description = description

    async def execute(self, input_data: Dict[str, Any]) -> str:
        """Execute the tool with the given input."""
        raise NotImplementedError


class ToolRegistry:
    """Registry for managing tools."""

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys()) 