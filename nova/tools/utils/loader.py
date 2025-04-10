import importlib.util
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Type

from nova.tools.base.tool import BaseTool, ToolConfig
from nova.tools.utils.config import ToolConfigManager

logger = logging.getLogger(__name__)


class DependencyError(Exception):
    """Error raised when tool dependencies cannot be satisfied."""
    pass


class ToolLoader:
    """Dynamic tool loader with dependency management."""

    def __init__(self, config_manager: Optional[ToolConfigManager] = None) -> None:
        """Initialize the tool loader.
        
        Args:
            config_manager: Optional configuration manager for tool configs
        """
        self.config_manager = config_manager or ToolConfigManager()
        self._loaded_tools: Dict[str, Type[BaseTool]] = {}
        self._tool_paths: Dict[str, str] = {}

    def discover_and_load_tools(self) -> Dict[str, Type[BaseTool]]:
        """Discover and load all available tools.
        
        Returns:
            Dictionary of tool names to tool classes
        """
        import os
        from pathlib import Path
        
        # Get the Nova package base directory
        base_dir = Path(__file__).parent.parent.parent.parent
        tools_dir = base_dir / "nova" / "tools"
        
        # Discover tools in the tools directory
        tool_paths = self.discover_tools(str(tools_dir))
        
        # Load each discovered tool
        for path in tool_paths:
            self.load_tool(path, str(base_dir))
        
        return self._loaded_tools

    def discover_tools(self, directory: str) -> List[str]:
        """Discover available tools in a directory.
        
        Args:
            directory: Directory to search for tools
            
        Returns:
            List of discovered tool paths
        """
        tool_paths: List[str] = []
        base_path = Path(directory)
        
        if not base_path.exists():
            logger.warning(f"Directory not found: {directory}")
            return tool_paths
        
        for file in base_path.rglob("*.py"):
            if file.name.startswith("_"):
                continue
            
            rel_path = file.relative_to(base_path)
            module_path = str(rel_path.with_suffix("")).replace("/", ".")
            tool_paths.append(module_path)
            
        return tool_paths

    def load_tool(self, module_path: str, base_path: str) -> Optional[Type[BaseTool]]:
        """Load a tool from a module path.
        
        Args:
            module_path: Path to the module containing the tool
            base_path: Base directory for module resolution
            
        Returns:
            Tool class if found and loaded successfully
        """
        try:
            # Convert module path to absolute path
            base = Path(base_path)
            module_parts = module_path.split(".")
            
            # Construct the full path to the module
            if module_parts[0] == "nova":
                # Handle nova package modules
                full_path = base.joinpath(*module_parts).with_suffix(".py")
            else:
                # Handle tools package modules
                full_path = base.joinpath("tools", *module_parts).with_suffix(".py")
            
            if not full_path.exists():
                logger.error(f"Module not found: {full_path}")
                return None
            
            # Load module
            spec = importlib.util.spec_from_file_location(module_path, str(full_path))
            if not spec or not spec.loader:
                logger.error(f"Failed to create module spec: {module_path}")
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_path] = module
            spec.loader.exec_module(module)
            
            # Find tool class in module
            for item in dir(module):
                obj = getattr(module, item)
                if (isinstance(obj, type) and 
                    issubclass(obj, BaseTool) and 
                    obj != BaseTool):
                    self._loaded_tools[obj.get_default_config().name] = obj
                    self._tool_paths[obj.get_default_config().name] = str(full_path)
                    return obj
            
            logger.warning(f"No tool class found in module: {module_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading tool {module_path}: {str(e)}")
            return None

    def get_tool_dependencies(self, tool_name: str) -> Set[str]:
        """Get the dependencies for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Set of dependency tool names
        """
        tool_class = self._loaded_tools.get(tool_name)
        if not tool_class:
            return set()
        
        config = tool_class.get_default_config()
        return set(config.dependencies)

    def resolve_dependencies(self, tool_name: str, visited: Optional[Set[str]] = None) -> List[str]:
        """Resolve tool dependencies in correct loading order.
        
        Args:
            tool_name: Name of the tool
            visited: Set of visited tools (for cycle detection)
            
        Returns:
            List of tool names in dependency order
            
        Raises:
            DependencyError: If dependencies cannot be resolved
        """
        if visited is None:
            visited = set()
            
        if tool_name in visited:
            raise DependencyError(f"Circular dependency detected for tool: {tool_name}")
            
        visited.add(tool_name)
        
        if tool_name not in self._loaded_tools:
            raise DependencyError(f"Tool not found: {tool_name}")
            
        result: List[str] = []
        for dep in self.get_tool_dependencies(tool_name):
            if dep not in self._loaded_tools:
                raise DependencyError(f"Dependency not found: {dep} (required by {tool_name})")
            result.extend(self.resolve_dependencies(dep, visited))
            
        result.append(tool_name)
        return result

    def load_with_dependencies(self, tool_name: str) -> List[Type[BaseTool]]:
        """Load a tool and its dependencies.
        
        Args:
            tool_name: Name of the tool to load
            
        Returns:
            List of loaded tool classes in dependency order
            
        Raises:
            DependencyError: If dependencies cannot be resolved
        """
        if tool_name not in self._loaded_tools:
            raise DependencyError(f"Tool not found: {tool_name}")
            
        dependency_order = self.resolve_dependencies(tool_name)
        return [self._loaded_tools[name] for name in dependency_order]

    def get_tool_info(self, tool_name: str) -> Dict[str, str]:
        """Get information about a loaded tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary containing tool information
        """
        if tool_name not in self._loaded_tools:
            return {}
            
        tool_class = self._loaded_tools[tool_name]
        config = tool_class.get_default_config()
        
        return {
            "name": config.name,
            "version": config.version,
            "description": config.description,
            "path": self._tool_paths.get(tool_name, ""),
            "dependencies": ", ".join(config.dependencies),
        } 