Memory System
============

The memory system in Nova provides context-aware storage and retrieval capabilities for the agent. It maintains both short-term and long-term memory, allowing the agent to learn from past experiences and make better decisions.

Overview
--------

The memory system consists of the following key components:

- **Memory Storage**: Stores task execution history with timestamps
- **Context Retrieval**: Retrieves relevant context based on current task
- **Memory Summarization**: Maintains summaries of task executions
- **Memory Management**: Handles memory size limits and cleanup

Usage
-----

Basic Usage
~~~~~~~~~~

.. code-block:: python

    from nova.core.memory import Memory

    # Initialize memory
    memory = Memory(max_entries=1000)  # Optional: set maximum number of entries

    # Add a memory entry
    await memory.add(
        task="Navigate to website",
        step={"type": "browser", "action": {"type": "navigate", "url": "https://example.com"}},
        result="Successfully navigated to example.com"
    )

    # Get context for a task
    context = await memory.get_context("Navigate to website")
    print(context)

    # Get summary of memories
    summary = await memory.get_summary()
    print(summary)

Memory Management
~~~~~~~~~~~~~~~

The memory system automatically manages memory size:

.. code-block:: python

    # Memory will automatically remove oldest entries when limit is reached
    memory = Memory(max_entries=2)
    await memory.add("task1", {}, "result1")
    await memory.add("task2", {}, "result2")
    await memory.add("task3", {}, "result3")  # task1 will be removed

Serialization
~~~~~~~~~~~~

Memories can be serialized to JSON for storage:

.. code-block:: python

    # Save memory to JSON
    json_str = memory.to_json()

    # Load memory from JSON
    new_memory = Memory.from_json(json_str)

API Reference
------------

Memory
~~~~~~

.. autoclass:: nova.core.memory.Memory
   :members:
   :undoc-members:
   :show-inheritance:

Memory Entry Format
-----------------

Each memory entry contains the following fields:

- **task**: Description of the task
- **step**: The step that was executed (browser action or tool action)
- **result**: The result of the step execution
- **timestamp**: ISO format timestamp of when the entry was created

Example:

.. code-block:: json

    {
        "task": "Navigate to website",
        "step": {
            "type": "browser",
            "action": {
                "type": "navigate",
                "url": "https://example.com"
            }
        },
        "result": "Successfully navigated to example.com",
        "timestamp": "2024-04-10T12:00:00.000Z"
    }

Context Retrieval
---------------

The memory system uses the following strategies to retrieve relevant context:

1. **Direct Task Match**: Memories with the same task description
2. **Keyword Matching**: Memories with similar keywords in the task description
3. **Recent Memories**: Most recent memories if no direct matches are found

The context is formatted as a natural language string combining relevant memories.

Memory Summarization
------------------

The memory system maintains summaries for each task type. Summaries are updated whenever a new memory is added for a task. The summary format is:

.. code-block:: text

    Task: <task description>
    Summary: <combined results of all executions of this task>

Best Practices
-------------

1. **Memory Size**: Set appropriate `max_entries` based on your use case
2. **Task Descriptions**: Use consistent and descriptive task names
3. **Result Format**: Keep results concise and informative
4. **Regular Cleanup**: Use `clear()` when starting new sessions
5. **Serialization**: Save important memories to disk when needed 