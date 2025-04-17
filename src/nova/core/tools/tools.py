from typing import Any, Dict, List, Optional, Type, Union, Callable
from dataclasses import dataclass
import json
import logging
from langchain.tools import BaseTool
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, create_model
import inspect

from .tool import ToolResult

logger = logging.getLogger(__name__)

@dataclass
class ToolConfig:
    """Configuration for a tool."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    func: Optional[Callable] = None
    is_async: bool = False
    requires_auth: bool = False
    auth_config: Optional[Dict[str, Any]] = None

class ToolInputModel(BaseModel):
    """Base model for tool input validation."""
    pass

class ToolOutputModel(BaseModel):
    """Base model for tool output validation."""
    pass

class Tool:
    """Base tool class integrating with LangChain."""
    
    def __init__(self, config: ToolConfig):
        self.config = config
        
    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """Execute the tool with the given input."""
        raise NotImplementedError("Subclasses must implement execute")
        
    def to_langchain_tool(self) -> BaseTool:
        """Convert this tool to a LangChain tool."""
        return BaseTool(
            name=self.config.name,
            description=self.config.description,
            func=self._execute_wrapper,
            args_schema=self.config.input_schema,
            return_schema=self.config.output_schema,
            handle_tool_error=True
        )
        
    async def _execute_wrapper(self, **kwargs) -> Any:
        """Wrapper for LangChain tool execution."""
        try:
            result = await self.execute(kwargs)
            if result.error:
                raise ToolException(result.error)
            return result.data
        except Exception as e:
            raise ToolException(str(e))

class ToolRegistry:
    """Registry for managing tools available to agents."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}  # Changed from ToolConfig to Tool
        self._input_models: Dict[str, Type[BaseModel]] = {}
        self._output_models: Dict[str, Type[BaseModel]] = {}
        
    def register_tool(self, tool: Tool) -> None:
        """Register a new tool instance."""
        if tool.config.name in self.tools:
            raise ValueError(f"Tool '{tool.config.name}' already registered")
            
        # Create input and output models
        input_model = self._create_model_from_schema(
            f"{tool.config.name}Input",
            tool.config.input_schema
        )
        output_model = self._create_model_from_schema(
            f"{tool.config.name}Output",
            tool.config.output_schema
        )
        
        # Store models and tool
        self._input_models[tool.config.name] = input_model
        self._output_models[tool.config.name] = output_model
        self.tools[tool.config.name] = tool
        
        # If the tool has sub-tools (like BrowserTools), register them too
        if hasattr(tool, 'tools') and isinstance(tool.tools, dict):
            for subtool_name, subtool_config in tool.tools.items():
                if subtool_name in self.tools:
                    logger.warning(f"Tool '{subtool_name}' already registered, skipping.")
                    continue
                
                # Create a Tool instance for the sub-tool
                class SubTool(Tool):
                    def __init__(self, config: ToolConfig):
                        super().__init__(config)
                        
                    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
                        if self.config.func is None:
                            raise ValueError(f"Tool '{self.config.name}' has no implementation")
                        if self.config.is_async:
                            result = await self.config.func(**input_data)
                        else:
                            result = self.config.func(**input_data)
                        return result
                
                subtool = SubTool(subtool_config)
                
                # Create input and output models for the subtool
                subtool_input_model = self._create_model_from_schema(
                    f"{subtool_name}Input",
                    subtool_config.input_schema
                )
                subtool_output_model = self._create_model_from_schema(
                    f"{subtool_name}Output",
                    subtool_config.output_schema
                )
                
                # Store models and tool
                self._input_models[subtool_name] = subtool_input_model
                self._output_models[subtool_name] = subtool_output_model
                self.tools[subtool_name] = subtool
        
        logger.info(f"Registered tool: {tool.config.name}")
        
    def _create_model_from_schema(
        self,
        model_name: str,
        schema: Dict[str, Any]
    ) -> Type[BaseModel]:
        """Create a Pydantic model from a JSON schema."""
        fields = {}
        
        if "properties" in schema:
            for field_name, field_schema in schema["properties"].items():
                field_type = self._get_field_type(field_schema)
                fields[field_name] = (
                    field_type,
                    Field(
                        description=field_schema.get("description", ""),
                        default=field_schema.get("default", ...)
                    )
                )
                
        return create_model(model_name, **fields)
        
    def _get_field_type(self, schema: Dict[str, Any]) -> Any:
        """Get Python type from JSON schema type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": List,
            "object": Dict[str, Any]
        }
        
        if "type" in schema:
            base_type = type_map.get(schema["type"])
            if base_type is None:
                raise ValueError(f"Unsupported type: {schema['type']}")
                
            if schema["type"] == "array" and "items" in schema:
                item_type = self._get_field_type(schema["items"])
                return List[item_type]
                
            return base_type
            
        return Any
        
    def get_tool_functions(self) -> Dict[str, Dict[str, Any]]:
        """Get descriptions of all registered tools."""
        return {
            name: {
                "description": tool.config.description,
                "input_schema": tool.config.input_schema,
                "output_schema": tool.config.output_schema,
                "is_async": tool.config.is_async,
                "requires_auth": tool.config.requires_auth
            }
            for name, tool in self.tools.items()
        }
        
    async def execute_tool(
        self,
        tool_name: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool with input validation."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")
            
        tool = self.tools[tool_name]
        input_model = self._input_models[tool_name]
        output_model = self._output_models[tool_name]
        
        try:
            # Validate input
            validated_input = input_model(**input_data)
            
            # Execute tool
            if tool.config.func is None:
                raise ValueError(f"Tool '{tool_name}' has no implementation")
                
            if tool.config.is_async:
                result = await tool.config.func(**validated_input.dict())
            else:
                result = tool.config.func(**validated_input.dict())
                
            # Convert ToolResult to dict if needed
            if hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
            else:
                result_dict = result
                
            # Validate output
            validated_output = output_model(**result_dict)
            return validated_output.dict()
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}", exc_info=True)
            raise
            
    def get_langchain_tools(self) -> List[BaseTool]:
        """Convert registered tools to LangChain tools."""
        tools = []
        
        for name, tool in self.tools.items():
            if tool.config.func is None:
                continue
                
            # Create tool class
            class DynamicTool(BaseTool):
                name = name
                description = tool.config.description
                
                def _run(self, **kwargs: Any) -> Any:
                    return self._execute_tool(kwargs)
                    
                async def _arun(self, **kwargs: Any) -> Any:
                    return await self._execute_tool(kwargs)
                    
                def _execute_tool(self, kwargs: Dict[str, Any]) -> Any:
                    if tool.config.is_async:
                        raise NotImplementedError(
                            "Async tools must be called with _arun"
                        )
                    return tool.config.func(**kwargs)
                    
            tools.append(DynamicTool())
            
        return tools
        
    def clear(self) -> None:
        """Clear all registered tools."""
        self.tools.clear()
        self._input_models.clear()
        self._output_models.clear()
        
    def get_tool_by_type(self, tool_type: Type[Tool]) -> Optional[Tool]:
        """Get a tool instance by its type.
        
        Args:
            tool_type: The type of tool to find
            
        Returns:
            The tool instance if found, None otherwise
        """
        for tool in self.tools.values():
            if isinstance(tool, tool_type):
                return tool
        return None

    def get_tool_names(self) -> List[str]:
        """Get a list of all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys()) 