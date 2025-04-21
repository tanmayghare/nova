import logging
from typing import Any, Dict, List, Optional, Type, Callable
from dataclasses import dataclass
from langchain.tools import BaseTool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field, create_model, ValidationError

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
        required_fields = schema.get("required", []) # Get the list of required fields
        
        if "properties" in schema:
            for field_name, field_schema in schema["properties"].items():
                field_type = self._get_field_type(field_schema)
                
                # Determine the default value based on whether the field is required
                if field_name in required_fields:
                    field_default = ... # Ellipsis marks field as required by Pydantic
                else:
                    field_default = None # None marks field as optional by Pydantic
                    
                fields[field_name] = (
                    field_type,
                    Field(
                        description=field_schema.get("description", ""),
                        # Use the determined default value
                        default=field_default
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
            "object": Dict[str, Any],
            "null": type(None)
        }

        schema_type = schema.get("type")

        if isinstance(schema_type, list):
            is_optional = "null" in schema_type
            non_null_types = [t for t in schema_type if t != "null"]

            if not non_null_types:
                return type(None)
                
            if len(non_null_types) > 1:
                logger.warning(f"Multiple non-null types in schema: {non_null_types}. Using first: {non_null_types[0]}")
                base_type_str = non_null_types[0]
            else:
                base_type_str = non_null_types[0]

            base_schema_for_type = schema.copy()
            base_schema_for_type["type"] = base_type_str
            resolved_type = self._get_field_type(base_schema_for_type)
            
            return Optional[resolved_type] if is_optional else resolved_type

        elif isinstance(schema_type, str):
            base_type = type_map.get(schema_type)
            if base_type is None:
                raise ValueError(f"Unsupported type string: {schema_type}")

            if schema_type == "array" and "items" in schema:
                item_type = self._get_field_type(schema["items"])
                return List[item_type]
            elif schema_type == "object":
                return Dict[str, Any]
            else:
                return base_type
                
        elif schema_type is None:
            logger.warning(f"No 'type' key found in schema field: {schema}. Defaulting to Any.")
            return Any
        else:
             raise ValueError(f"Invalid schema type value: {schema_type}")
        
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
            # Check if the tool object itself has an execute method (preferred)
            if hasattr(tool, 'execute') and callable(tool.execute):
                 logger.debug(f"Executing tool '{tool_name}' via its execute() method.")
                 # Assuming the execute method handles async internally if needed
                 # and expects the raw input dict (or validated Pydantic model)
                 # Let's pass the validated model dump for consistency
                 result = await tool.execute(validated_input.model_dump()) 
            # Fallback: Execute via function specified in config.func
            elif tool.config.func is not None:
                 logger.debug(f"Executing tool '{tool_name}' via config.func.")
                 if tool.config.is_async:
                      result = await tool.config.func(**validated_input.model_dump())
                 else:
                      result = tool.config.func(**validated_input.model_dump())
            else:
                 # If neither execute() nor config.func exists, then no implementation
                 raise ValueError(f"Tool '{tool_name}' has no implementation (no execute method or config.func)")

            # Prepare dictionary for output validation
            output_data_to_validate = {}
            if isinstance(result, ToolResult): 
                # If ToolResult, validate its DATA field against the output schema
                output_data_to_validate = result.data if result.data is not None else {}
            elif isinstance(result, dict):
                # If raw dict returned (e.g., from config.func), validate it directly
                output_data_to_validate = result
            else:
                # Handle non-dict/non-ToolResult returns
                logger.warning(f"Tool '{tool_name}' returned unexpected type {type(result)}. Wrapping in data dict for validation.")
                # Attempt a basic wrapping, might fail validation if schema is complex
                output_data_to_validate = {"success": True, "data": result} # Wrap based on schema?

            # Validate output against the defined output_schema
            logger.debug(f"Validating output data for tool '{tool_name}': {output_data_to_validate}")
            validated_output = output_model(**output_data_to_validate)
            
            # If the original result was a ToolResult, return its full dictionary representation.
            # Otherwise, assume the raw dict result IS the final structure.
            if isinstance(result, ToolResult):
                # Update the data field with the validated version, just in case validation did coercion
                result.data = validated_output.model_dump()
                return result.to_dict() # Return the whole ToolResult structure
            else:
                 # Assume the raw dict (output_data_to_validate) passed validation and is the intended full result.
                 # Re-validate the whole thing? Or trust it? Let's return it as is.
                 # If it was a raw dict, validated_output is that dict. 
                 # Return the result that passed validation.
                 return validated_output.model_dump()
            
        except ValidationError as ve:
            logger.error(f"Tool '{tool_name}' output validation failed: {ve}", exc_info=True)
            # Construct a standard error ToolResult dict to return
            error_detail = f"Output validation failed: {ve}"
            # If original result was ToolResult and had an error, prioritize that?
            original_error = getattr(result, 'error', None) if isinstance(result, ToolResult) else None
            if original_error:
                 error_detail = f"Original tool error: {original_error}; Output validation failed: {ve}"
            # IMPORTANT: We need to return a dict here, not raise, so the agent can handle it.
            # Return a structure indicating failure, matching expected error format if possible.
            return {"success": False, "status": "error", "error": error_detail} 
        except Exception as e:
            logger.error(f"Tool '{tool_name}' execution failed: {e}", exc_info=True)
            # IMPORTANT: Return a dict here, not raise.
            return {"success": False, "status": "error", "error": f"Tool execution failed: {str(e)}"}
            
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