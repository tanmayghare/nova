from typing import Any, Dict, List, Optional

from ..tools.base.tool import ToolResult


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

    async def execute_tool(self, name: str, input_data: Dict[str, Any]) -> ToolResult:
        """Execute a tool with the given input.
        
        Args:
            name: Name of the tool to execute
            input_data: Input parameters for the tool
            
        Returns:
            ToolResult containing execution results
        """
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool not found: {name}",
                execution_time=0.0
            )
        
        try:
            # Ensure input parameters are properly formatted
            cleaned_input = {}
            for key, value in input_data.items():
                if isinstance(value, str):
                    value = value.strip('"\'')  # Remove any quotes
                cleaned_input[key] = value
            
            return await tool.execute(cleaned_input)
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time=0.0
            ) 