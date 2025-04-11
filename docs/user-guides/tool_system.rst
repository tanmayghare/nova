Tool System
===========

The Nova tool system provides a powerful framework for creating, managing, and executing tools. This document covers the core concepts, advanced features, and best practices for working with the tool system.

Core Concepts
------------

Base Tool
~~~~~~~~

The ``BaseTool`` class is the foundation of the tool system. All tools must inherit from this class and implement the required methods:

.. code-block:: python

    from nova.tools import BaseTool, ToolResult

    class MyTool(BaseTool):
        @classmethod
        def get_default_config(cls):
            return ToolConfig(
                name="my_tool",
                version="1.0.0",
                description="My custom tool"
            )

        async def validate_input(self, input_data):
            return True

        async def execute(self, input_data):
            # Tool implementation
            return ToolResult(success=True, data=result)

Tool Registry
~~~~~~~~~~~~

The ``ToolRegistry`` manages tool registration and discovery:

.. code-block:: python

    from nova.tools import ToolRegistry

    registry = ToolRegistry()
    registry.register(MyTool())

Advanced Features
---------------

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~

The tool system includes built-in performance monitoring:

.. code-block:: python

    # Get metrics for a tool
    metrics = tool.get_metrics()
    avg_time = metrics.get_average_execution_time("tool_name")
    success_rate = metrics.get_success_rate("tool_name")

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~

Tools can be configured with advanced settings:

.. code-block:: python

    config = AdvancedToolConfig(
        name="my_tool",
        version="2.0.0",
        timeout=30,
        max_memory_mb=100,
        log_level="INFO",
        custom_settings={"setting": "value"}
    )

User-Defined Tools
~~~~~~~~~~~~~~~~~

Users can create their own tools:

.. code-block:: python

    from nova.tools.utils.user_tools import UserToolManager

    manager = UserToolManager()
    manager.create_tool_template("my_tool")
    # Edit the generated template
    tool = manager.load_user_tool("my_tool")

Best Practices
-------------

1. Input Validation
   - Always validate input data in the ``validate_input`` method
   - Return clear error messages for invalid input

2. Error Handling
   - Use ``ToolResult`` to return execution results
   - Include detailed error messages
   - Track execution time

3. Performance
   - Monitor execution time
   - Use appropriate timeouts
   - Implement caching where appropriate

4. Configuration
   - Use meaningful default values
   - Document configuration options
   - Validate configuration values

Examples
--------

See the ``examples`` directory for complete examples:

- ``tool_system_test.py``: Basic tool usage
- ``advanced_features_test.py``: Advanced features
- ``tool_chain_test.py``: Tool chaining 