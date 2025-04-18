import time
from typing import Any, Dict

from nova.core.tools.tool import BaseTool, ToolConfig, ToolResult


class ExampleAPITool(BaseTool):
    """Example API tool that demonstrates the enhanced tool system."""

    @classmethod
    def get_default_config(cls) -> ToolConfig:
        """Get the default configuration for this tool."""
        return ToolConfig(
            name="example_api",
            version="1.0.0",
            description="Example API tool that demonstrates the enhanced tool system",
            enabled=True,
            timeout=30,
            retry_attempts=3,
            dependencies=["requests"]
        )

    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate the input data.
        
        Args:
            input_data: Dictionary containing input parameters
            
        Returns:
            bool indicating whether the input is valid
        """
        required_fields = ["endpoint", "method"]
        return all(field in input_data for field in required_fields)

    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """Execute the API request.
        
        Args:
            input_data: Dictionary containing input parameters
            
        Returns:
            ToolResult containing execution results
        """
        start_time = time.time()
        
        try:
            # Simulate API request
            endpoint = input_data["endpoint"]
            method = input_data["method"]
            params = input_data.get("params", {})
            
            # In a real implementation, this would make an actual API request
            # For now, we'll just simulate a successful response
            response = {
                "status": "success",
                "data": {
                    "endpoint": endpoint,
                    "method": method,
                    "params": params
                }
            }
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                success=True,
                data=response,
                error=None,
                execution_time=execution_time,
                metadata={
                    "endpoint": endpoint,
                    "method": method,
                    "params": params
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time=execution_time
            )

    async def cleanup(self) -> None:
        """Clean up any resources."""
        # In a real implementation, this would close any open connections
        pass 