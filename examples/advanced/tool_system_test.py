"""Test the enhanced tool system."""

import asyncio
import time
from typing import Any, Dict

from nova.tools import (
    BaseTool,
    ToolResult,
    ToolRegistry,
)
from nova.tools.utils.config import AdvancedToolConfig


class CalculatorTool(BaseTool):
    """A simple calculator tool."""

    @classmethod
    def get_default_config(cls) -> AdvancedToolConfig:
        return AdvancedToolConfig(
            name="calculator",
            version="1.0.0",
            description="A simple calculator tool",
            enabled=True,
            timeout=10,
            retry_attempts=2,
            max_memory_mb=50,
            log_level="INFO",
            cache_enabled=False,
            concurrent_executions=1,
            custom_settings={"precision": 2}
        )

    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        required_fields = ["operation", "a", "b"]
        if not all(field in input_data for field in required_fields):
            return False
            
        valid_operations = {"add", "subtract", "multiply", "divide"}
        if input_data["operation"] not in valid_operations:
            return False
            
        try:
            float(input_data["a"])
            float(input_data["b"])
        except (TypeError, ValueError):
            return False
            
        if input_data["operation"] == "divide" and float(input_data["b"]) == 0:
            return False
            
        return True

    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """Execute the calculator operation.
        
        Args:
            input_data: Dictionary containing:
                - operation: The operation to perform
                - a: First number
                - b: Second number
            
        Returns:
            ToolResult containing the calculation result
        """
        try:
            operation = input_data["operation"]
            a = float(input_data["a"])
            b = float(input_data["b"])
            
            start_time = time.time()
            
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    raise ValueError("Division by zero")
                result = a / b
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            # Apply precision from custom settings
            precision = self.config.custom_settings.get("precision", 2)
            result = round(result, precision)
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                success=True,
                data={"result": result},
                error=None,
                execution_time=execution_time,
                metadata={
                    "operation": operation,
                    "a": a,
                    "b": b,
                    "precision": precision
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                execution_time=time.time() - start_time
            )


async def main() -> None:
    """Test the calculator tool."""
    # Initialize components
    registry = ToolRegistry()
    
    # Register calculator
    calculator = CalculatorTool()
    registry.register(calculator)
    
    # Test calculations
    operations = [
        {"operation": "add", "a": 5, "b": 3},
        {"operation": "multiply", "a": 4, "b": 2},
        {"operation": "divide", "a": 10, "b": 2},
        {"operation": "divide", "a": 10, "b": 0},  # Should fail
        {"operation": "unknown", "a": 1, "b": 1},  # Should fail
    ]
    
    print("\n=== Testing Calculator Tool ===")
    for op in operations:
        result = await calculator.run(op)
        status = "✓" if result.success else "✗"
        print(f"\n{status} Operation: {op}")
        if result.success:
            print(f"Result: {result.data}")
        else:
            print(f"Error: {result.error}")
        print(f"Execution time: {result.execution_time:.3f}s")
    
    # Print metrics
    metrics = calculator.get_metrics()
    print("\nCalculator Metrics:")
    print(f"Average execution time: {metrics.get_average_execution_time('calculator'):.3f}s")
    print(f"Success rate: {metrics.get_success_rate('calculator'):.1f}%")


if __name__ == "__main__":
    asyncio.run(main()) 