Tool System
===========

The Nova tool system provides a flexible framework for creating and executing browser automation tools using LangChain. This document covers the core concepts, browser tools, and best practices for working with the tool system.

Core Concepts
------------

Browser Tools
~~~~~~~~~~~~

The browser tools are implemented using LangChain's BaseTool class:

.. code-block:: python

    from nova.tools.browser import get_browser_tools
    from nova.core.browser import Browser

    # Initialize browser
    browser = Browser(
        headless=False,
        timeout=30,
        viewport={"width": 1280, "height": 720}
    )

    # Get browser tools
    tools = get_browser_tools(browser)

    # Use tools
    await tools[0]._arun(url="https://example.com")  # Navigate
    await tools[1]._arun(selector="#button")  # Click
    await tools[2]._arun(selector="#input", text="Hello, World!")  # Type

Tool Registration
~~~~~~~~~~~~~~~

Tools are registered with the agent during initialization:

.. code-block:: python

    from nova.agents.task.task_agent import TaskAgent
    from nova.core.memory import Memory

    # Initialize components
    memory = Memory()
    browser = Browser()
    tools = get_browser_tools(browser)

    # Create agent with tools
    agent = TaskAgent(
        llm_config=llm_config,
        memory=memory,
        tools=tools
    )

Browser Actions
-------------

The following browser actions are supported:

Navigation
~~~~~~~~~

.. code-block:: python

    # Navigate to URL
    await tools[0]._arun(url="https://example.com")

Element Interaction
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Click element
    await tools[1]._arun(selector="#button")

    # Type text
    await tools[2]._arun(selector="#input", text="Hello, World!")

    # Get text
    text = await tools[3]._arun(selector="#content")

    # Get HTML
    html = await tools[4]._arun()

    # Take screenshot
    await tools[5]._arun(path="screenshot.png")

    # Wait for element
    await tools[6]._arun(selector="#loading", timeout=10)

    # Scroll page
    await tools[7]._arun(direction="down")

Content Retrieval
~~~~~~~~~~~~~~~

.. code-block:: python

    # Get text
    text = await tools.get_text("#content")

    # Get attribute
    href = await tools.get_attribute("a", "href")

    # Get all matching elements
    elements = await tools.get_elements(".item")

    # Check if element exists
    exists = await tools.element_exists("#element")

Screenshot and DOM
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Take screenshot
    await tools.screenshot("screenshot.png")

    # Get page source
    source = await tools.get_page_source()

    # Get DOM snapshot
    dom = await tools.get_dom_snapshot()

Error Handling
------------

The tool system includes comprehensive error handling:

.. code-block:: python

    try:
        await tools.click("#missing-button")
    except Exception as e:
        print(f"Error: {e}")
        # Handle error

Performance Monitoring
--------------------

Track tool performance:

.. code-block:: python

    # Get tool metrics
    metrics = tools.get_metrics()
    print(f"Total actions: {metrics.total_actions}")
    print(f"Success rate: {metrics.success_rate}")
    print(f"Average duration: {metrics.average_duration}")

Best Practices
-------------

1. **Element Selection**:
   - Use unique and stable selectors
   - Prefer IDs over classes
   - Use text content as fallback

2. **Error Handling**:
   - Always handle potential errors
   - Provide meaningful error messages
   - Implement retry logic where appropriate

3. **Performance**:
   - Minimize unnecessary actions
   - Use appropriate timeouts
   - Monitor execution time

4. **State Management**:
   - Update memory after significant actions
   - Track success/failure of actions
   - Maintain context for error recovery

Examples
--------

See the ``examples`` directory for complete examples:

- ``browser_automation.py``: Basic browser automation
- ``form_filling.py``: Form interaction
- ``scraping.py``: Content extraction 