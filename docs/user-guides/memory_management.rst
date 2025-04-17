Memory System
============

The memory system in Nova provides context-aware storage and retrieval capabilities for the agent. It maintains both short-term and long-term memory, allowing the agent to learn from past experiences and make better decisions.

Overview
--------

The memory system consists of the following key components:

- **State Storage**: Stores agent state and execution history
- **Context Management**: Manages task context and relevant information
- **Performance Tracking**: Tracks execution metrics and performance
- **Error Recovery**: Maintains error history for improved handling

Usage
-----

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from nova.core.memory import Memory

    # Initialize memory
    memory = Memory()

    # Update state
    memory.update({
        "current_url": "https://example.com",
        "last_action": "click",
        "timestamp": "2024-04-17T12:00:00Z"
    })

    # Get relevant state
    relevant = memory.get_relevant({
        "current_url": "https://example.com"
    })
    print(relevant)

    # Track performance
    memory.track_performance({
        "action": "navigate",
        "duration": 1.5,
        "success": True
    })

State Management
~~~~~~~~~~~~~~

The memory system manages agent state:

.. code-block:: python

    # Store complete state
    state = {
        "url": "https://example.com",
        "actions": ["navigate", "click"],
        "timestamp": "2024-04-17T12:00:00Z",
        "performance": {
            "total_actions": 2,
            "success_rate": 1.0
        }
    }
    memory.update(state)

    # Retrieve state
    current_state = memory.get_state()
    print(current_state)

Performance Tracking
~~~~~~~~~~~~~~~~~

Track execution metrics:

.. code-block:: python

    # Track action performance
    memory.track_performance({
        "action": "navigate",
        "duration": 1.5,
        "success": True,
        "error": None
    })

    # Track LLM performance
    memory.track_performance({
        "component": "llm",
        "duration": 0.8,
        "tokens_used": 150,
        "success": True
    })

    # Get performance metrics
    metrics = memory.get_performance_metrics()
    print(metrics)

Error Handling
~~~~~~~~~~~~

Manage error history:

.. code-block:: python

    # Track error
    memory.track_error({
        "type": "browser_error",
        "message": "Element not found",
        "timestamp": "2024-04-17T12:00:00Z",
        "context": {
            "url": "https://example.com",
            "action": "click",
            "selector": "#missing-button"
        }
    })

    # Get error history
    errors = memory.get_error_history()
    print(errors)

API Reference
------------

Memory
~~~~~~

.. autoclass:: nova.core.memory.Memory
   :members:
   :undoc-members:
   :show-inheritance:

State Format
-----------

The memory system stores state in the following format:

.. code-block:: json

    {
        "current_url": "https://example.com",
        "last_action": "click",
        "timestamp": "2024-04-17T12:00:00Z",
        "performance": {
            "total_actions": 10,
            "success_rate": 0.9,
            "average_duration": 1.2
        },
        "errors": [
            {
                "type": "browser_error",
                "message": "Element not found",
                "timestamp": "2024-04-17T12:00:00Z"
            }
        ]
    }

Performance Metrics
-----------------

The memory system tracks the following metrics:

1. **Action Performance**:
   - Duration
   - Success rate
   - Error count
   - Average duration

2. **LLM Performance**:
   - Token usage
   - Response time
   - Success rate
   - Error rate

3. **Overall Metrics**:
   - Total actions
   - Total errors
   - Average performance
   - Success rate

Best Practices
-------------

1. **State Updates**: Update state after each significant action
2. **Performance Tracking**: Track all major operations
3. **Error Handling**: Log all errors with context
4. **State Relevance**: Use specific queries for relevant state
5. **Regular Cleanup**: Clear old state when starting new sessions 