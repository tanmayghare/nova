import asyncio
from pathlib import Path

from nova.tools import ToolRegistry
from nova.tools.utils.chain import ToolChain
from nova.tools.utils.loader import ToolLoader


async def test_tool_chain() -> None:
    """Test the tool chain functionality."""
    # Initialize components
    registry = ToolRegistry()
    chain = ToolChain(registry)
    
    # Register tools
    from examples.tool_system_test import CalculatorTool
    calculator = CalculatorTool()
    registry.register(calculator)
    
    # Create a chain that performs multiple calculations
    print("\n=== Testing Tool Chain ===")
    
    # Add steps to the chain
    chain.add_step(
        tool_name="calculator",
        input_data={"operation": "add", "a": 5, "b": 3},
        max_retries=2,
    )
    
    chain.add_step(
        tool_name="calculator",
        input_data={"operation": "multiply", "a": 2, "b": 4},
        depends_on=["calculator"],  # Depends on first calculation
    )
    
    chain.add_step(
        tool_name="calculator",
        input_data={"operation": "divide", "a": 10, "b": 0},  # This will fail
        max_retries=1,
    )
    
    # Execute the chain
    result = await chain.execute()
    
    # Print results
    print(f"\nChain execution {'succeeded' if result.success else 'failed'}")
    print(f"Total execution time: {result.total_execution_time:.2f} seconds")
    if result.error:
        print(f"Error: {result.error}")
    
    print("\nStep results:")
    for step in result.steps:
        status = "✓" if step["success"] else "✗"
        print(f"{status} {step['tool_name']}:")
        print(f"  Input: {step['input']}")
        print(f"  Output: {step['output']}")
        if step["error"]:
            print(f"  Error: {step['error']}")
        print(f"  Time: {step['execution_time']:.2f}s")


async def test_tool_loader() -> None:
    """Test the tool loader functionality."""
    print("\n=== Testing Tool Loader ===")
    
    # Initialize loader
    loader = ToolLoader()
    
    # Get the tools directory
    tools_dir = Path(__file__).parent.parent / "nova" / "tools"
    
    # Discover and load tools
    print("\nDiscovering tools...")
    tool_paths = loader.discover_tools(str(tools_dir))
    print(f"Found {len(tool_paths)} tool modules:")
    for path in tool_paths:
        print(f"- {path}")
    
    # Load tools
    print("\nLoading tools...")
    for path in tool_paths:
        tool_class = loader.load_tool(path, str(tools_dir.parent))
        if tool_class:
            config = tool_class.get_default_config()
            print(f"\nLoaded tool: {config.name}")
            print(f"Version: {config.version}")
            print(f"Description: {config.description}")
            if config.dependencies:
                print(f"Dependencies: {', '.join(config.dependencies)}")
    
    # Test dependency resolution
    print("\nTesting dependency resolution...")
    try:
        for tool_name in loader._loaded_tools:
            deps = loader.get_tool_dependencies(tool_name)
            if deps:
                print(f"\n{tool_name} dependencies: {deps}")
                resolved = loader.resolve_dependencies(tool_name)
                print(f"Resolved order: {' -> '.join(resolved)}")
    except Exception as e:
        print(f"Error resolving dependencies: {e}")


async def main() -> None:
    """Run all tests."""
    await test_tool_chain()
    await test_tool_loader()


if __name__ == "__main__":
    asyncio.run(main()) 