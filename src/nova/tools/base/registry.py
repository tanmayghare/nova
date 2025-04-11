import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from nova.tools.base.tool import BaseTool, ToolConfig, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Enhanced registry for managing tools with dynamic loading capabilities."""

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self._tool_configs: Dict[str, ToolConfig] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance.
        
        Args:
            tool: The tool instance to register
        """
        if not isinstance(tool, BaseTool):
            raise ValueError("Only BaseTool instances can be registered")
        
        self._tools[tool.config.name] = tool
        self._tool_configs[tool.config.name] = tool.config
        logger.info(f"Registered tool: {tool.config.name}")

    def register_class(self, tool_class: Type[BaseTool]) -> None:
        """Register a tool class.
        
        Args:
            tool_class: The tool class to register
        """
        if not inspect.isclass(tool_class) or not issubclass(tool_class, BaseTool):
            raise ValueError("Only BaseTool classes can be registered")
        
        config = tool_class.get_default_config()
        self._tool_classes[config.name] = tool_class
        self._tool_configs[config.name] = config
        logger.info(f"Registered tool class: {config.name}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool instance by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            The tool instance if found, None otherwise
        """
        return self._tools.get(name)

    def get_tool_class(self, name: str) -> Optional[Type[BaseTool]]:
        """Get a tool class by name.
        
        Args:
            name: Name of the tool class
            
        Returns:
            The tool class if found, None otherwise
        """
        return self._tool_classes.get(name)

    def create_tool(self, name: str, config: Optional[ToolConfig] = None) -> Optional[BaseTool]:
        """Create a new tool instance from a registered class.
        
        Args:
            name: Name of the tool class
            config: Optional configuration to override defaults
            
        Returns:
            New tool instance if class found, None otherwise
        """
        tool_class = self.get_tool_class(name)
        if not tool_class:
            return None
        
        return tool_class(config)

    def list_tools(self) -> List[str]:
        """List all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def list_tool_classes(self) -> List[str]:
        """List all registered tool class names.
        
        Returns:
            List of tool class names
        """
        return list(self._tool_classes.keys())

    def load_tools_from_directory(self, directory: str) -> None:
        """Load tools from a directory.
        
        Args:
            directory: Path to the directory containing tool modules
        """
        path = Path(directory)
        if not path.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        for file in path.glob("**/*.py"):
            if file.name.startswith("_"):
                continue
            
            module_path = str(file.relative_to(path.parent)).replace("/", ".")[:-3]
            try:
                module = importlib.import_module(module_path)
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BaseTool) and 
                        obj != BaseTool):
                        self.register_class(obj)
                        logger.info(f"Loaded tool class from {module_path}: {name}")
            except Exception as e:
                logger.error(f"Error loading module {module_path}: {str(e)}")

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
            if not await tool.validate_input(input_data):
                return ToolResult(
                    success=False,
                    data=None,
                    error="Invalid input data",
                    execution_time=0.0
                )
            
            return await tool.execute(input_data)
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time=0.0
            )

    def unregister(self, name: str) -> None:
        """Unregister a tool or tool class.
        
        Args:
            name: Name of the tool or tool class to unregister
        """
        if name in self._tools:
            del self._tools[name]
        if name in self._tool_classes:
            del self._tool_classes[name]
        if name in self._tool_configs:
            del self._tool_configs[name]
        logger.info(f"Unregistered tool: {name}") 