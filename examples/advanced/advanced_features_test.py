"""Test advanced features of the tool system."""

import asyncio
from pathlib import Path

from nova.tools.utils.config import AdvancedToolConfig, ConfigurationManager
from nova.tools.utils.user_tools import UserToolManager
from examples.tool_system_test import CalculatorTool


async def test_performance_monitoring() -> None:
    """Test performance monitoring features."""
    print("\n=== Testing Performance Monitoring ===")
    
    # Create and execute calculator tool
    calculator = CalculatorTool()
    
    # Execute multiple operations
    operations = [
        {"operation": "add", "a": 5, "b": 3},
        {"operation": "multiply", "a": 4, "b": 2},
        {"operation": "subtract", "a": 10, "b": 7},
    ]
    
    for op in operations:
        result = await calculator.run(op)
        print(f"\nOperation: {op}")
        print(f"Result: {result.data}")
        print(f"Execution time: {result.execution_time:.3f}s")
    
    # Get metrics
    metrics = calculator.get_metrics()
    avg_time = metrics.get_average_execution_time("calculator")
    success_rate = metrics.get_success_rate("calculator")
    
    print(f"\nCalculator metrics:")
    print(f"Average execution time: {avg_time:.3f}s")
    print(f"Success rate: {success_rate:.1f}%")


async def test_advanced_config() -> None:
    """Test advanced configuration features."""
    print("\n=== Testing Advanced Configuration ===")
    
    # Initialize configuration manager
    config_dir = Path.home() / ".nova/config"
    config_manager = ConfigurationManager(str(config_dir))
    
    # Create advanced configuration
    calculator_config = AdvancedToolConfig(
        name="calculator",
        version="2.0.0",
        description="Advanced calculator tool",
        enabled=True,
        timeout=5,
        retry_attempts=2,
        max_memory_mb=100,
        max_cpu_percent=50.0,
        log_level="DEBUG",
        cache_enabled=True,
        cache_ttl=1800,
        rate_limit=100,
        concurrent_executions=2,
        environment_vars={"PRECISION": "high"},
        custom_settings={"rounding_digits": 4}
    )
    
    # Save configuration
    config_manager.set_config(calculator_config)
    print(f"Saved calculator configuration")
    
    # Load and verify configuration
    loaded_config = config_manager.get_config("calculator")
    if loaded_config:
        print(f"\nLoaded configuration:")
        print(f"Name: {loaded_config.name}")
        print(f"Version: {loaded_config.version}")
        print(f"Timeout: {loaded_config.timeout}s")
        print(f"Memory limit: {loaded_config.max_memory_mb}MB")
        print(f"Custom settings: {loaded_config.custom_settings}")
    
    # Update configuration
    config_manager.update_config("calculator", {
        "timeout": 10,
        "log_level": "INFO"
    })
    print("\nUpdated configuration")
    
    # Verify updates
    updated_config = config_manager.get_config("calculator")
    if updated_config:
        print(f"New timeout: {updated_config.timeout}s")
        print(f"New log level: {updated_config.log_level}")


async def test_user_tools() -> None:
    """Test user-defined tools."""
    print("\n=== Testing User-Defined Tools ===")
    
    # Initialize user tool manager
    user_tools_dir = Path.home() / ".nova/tools"
    tool_manager = UserToolManager(str(user_tools_dir))
    
    # Create a template for a new tool
    tool_manager.create_tool_template("string_analyzer")
    print("\nCreated template for StringAnalyzer tool")
    
    # Discover user tools
    tool_paths = tool_manager.discover_user_tools()
    print(f"\nDiscovered user tools: {tool_paths}")
    
    # Load user tools
    for path in tool_paths:
        tool_class = tool_manager.load_user_tool(path)
        if tool_class:
            print(f"Loaded tool: {tool_class.get_default_config().name}")
    
    # List loaded tools
    user_tools = tool_manager.list_user_tools()
    print(f"\nLoaded user tools: {list(user_tools.keys())}")


async def main() -> None:
    """Run all advanced feature tests."""
    await test_performance_monitoring()
    await test_advanced_config()
    await test_user_tools()


if __name__ == "__main__":
    asyncio.run(main()) 