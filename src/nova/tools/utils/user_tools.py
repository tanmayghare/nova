"""Support for user-defined tools."""

import os
import sys
import importlib.util
from typing import Any, Dict, List, Optional
from pathlib import Path


class UserToolManager:
    """Manager for user-defined tools."""
    
    def __init__(self, user_tools_dir: Optional[str] = None) -> None:
        """Initialize the user tool manager.
        
        Args:
            user_tools_dir: Directory for user-defined tools
        """
        self.tools_dir = Path(user_tools_dir or os.path.expanduser("~/.nova/tools"))
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self._user_tools: Dict[str, Any] = {}
        
        # Add user tools directory to Python path
        sys.path.append(str(self.tools_dir))
    
    def discover_user_tools(self) -> List[str]:
        """Discover available user-defined tools.
        
        Returns:
            List of discovered tool paths
        """
        tool_paths = []
        for file in self.tools_dir.rglob("*.py"):
            if file.name.startswith("_"):
                continue
            rel_path = file.relative_to(self.tools_dir)
            module_path = str(rel_path.with_suffix("")).replace("/", ".")
            tool_paths.append(module_path)
        return tool_paths
    
    def load_user_tool(self, module_path: str) -> Optional[Any]:
        """Load a user-defined tool.
        
        Args:
            module_path: Path to the tool module
            
        Returns:
            Tool class if found and loaded successfully
        """
        try:
            # Import here to avoid circular imports
            from nova.tools.base.tool import BaseTool
            
            # Convert module path to absolute path
            full_path = self.tools_dir / Path(module_path.replace(".", "/")).with_suffix(".py")
            
            if not full_path.exists():
                print(f"Tool not found: {full_path}")
                return None
            
            # Load module
            spec = importlib.util.spec_from_file_location(module_path, str(full_path))
            if not spec or not spec.loader:
                print(f"Failed to create module spec: {module_path}")
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
                    self._user_tools[obj.get_default_config().name] = obj
                    return obj
            
            print(f"No tool class found in module: {module_path}")
            return None
            
        except Exception as e:
            print(f"Error loading tool {module_path}: {str(e)}")
            return None
    
    def create_tool_template(self, tool_name: str) -> None:
        """Create a template for a new user-defined tool.
        
        Args:
            tool_name: Name of the new tool
        """
        template = f'''"""User-defined tool: {tool_name}"""

from typing import Any, Dict
from nova.tools.base.tool import BaseTool, ToolResult
from nova.tools.utils.config import AdvancedToolConfig


class {tool_name.capitalize()}Tool(BaseTool):
    """A user-defined tool."""
    
    @classmethod
    def get_default_config(cls) -> AdvancedToolConfig:
        return AdvancedToolConfig(
            name="{tool_name}",
            version="1.0.0",
            description="A user-defined tool",
            enabled=True,
            timeout=30,
            retry_attempts=3,
        )
    
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        # Add your input validation logic here
        return True
    
    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """Execute the tool with the given input.
        
        Args:
            input_data: Dictionary containing input parameters
            
        Returns:
            ToolResult containing execution results
        """
        try:
            # Add your tool implementation here
            result = "Tool not implemented yet"
            
            return ToolResult(
                success=True,
                data=result,
                error=None,
                execution_time=0.0,
                metadata=dict()
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time=0.0
            )
'''
        
        tool_path = self.tools_dir / f"{tool_name}.py"
        with open(tool_path, 'w') as f:
            f.write(template)
        print(f"Created tool template at {tool_path}")
    
    def get_user_tool(self, tool_name: str) -> Optional[Any]:
        """Get a user-defined tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool class if found
        """
        return self._user_tools.get(tool_name)
    
    def list_user_tools(self) -> Dict[str, Any]:
        """List all loaded user-defined tools.
        
        Returns:
            Dictionary of tool names to tool classes
        """
        return self._user_tools.copy() 