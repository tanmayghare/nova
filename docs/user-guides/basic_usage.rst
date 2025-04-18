Usage
=====

Basic Usage
==========

This guide covers the basic usage of Nova's browser automation tools.

Getting Started
--------------

First, import the necessary components:

.. code-block:: python

   from nova.core.browser import Browser
   from nova.tools.browser import get_browser_tools
   from nova.agents.task.task_agent import TaskAgent
   from nova.core.config import LLMConfig, AgentConfig
   from nova.core.memory import Memory
   import os
   from dotenv import load_dotenv

   load_dotenv()

   # Configure LLM
   llm_config = LLMConfig(
       provider="openai",
       model_name="gpt-3.5-turbo",
       temperature=0.1,
       max_tokens=1000
   )

   # Initialize components
   browser = Browser(
       headless=False,
       timeout=30,
       viewport={"width": 1280, "height": 720}
   )

   # Get browser tools
   tools = get_browser_tools(browser)

   memory = Memory()
   config = AgentConfig(llm_config=llm_config)

   # Create task agent
   agent = TaskAgent(
       task_id="example-task",
       task_description="Navigate to example.com and click a button",
       llm_config=llm_config,
       memory=memory,
       tools=tools
   )

   # Run the task
   result = await agent.run()
   print(result)

Configuration
------------

You can configure the agent using the ``AgentConfig`` class:

.. code-block:: python

   from nova.core.config import AgentConfig, LLMConfig

   # LLM configuration
   llm_config = LLMConfig(
       provider="anthropic",
       model_name="claude-3-opus-20240229",
       temperature=0.1,
       max_tokens=2000
   )

   # Agent configuration
   config = AgentConfig(
       llm_config=llm_config,
       max_steps=50,
       timeout=300,
       verbose=True
   )

   agent = TaskAgent(
       llm_config=llm_config,
       memory=memory,
       config=config
   )

Browser Configuration
-------------------

You can configure the browser using the ``Browser`` class:

.. code-block:: python

   from nova.core.browser import Browser
   from nova.tools.browser import get_browser_tools

   # Custom browser configuration
   browser = Browser(
       headless=False,
       timeout=30,
       viewport={"width": 1280, "height": 720}
   )

   # Get browser tools
   tools = get_browser_tools(browser)

   # Initialize agent with custom components
   agent = TaskAgent(
       llm_config=llm_config,
       memory=memory,
       tools=tools
   )

Memory System
------------

Nova includes a memory system for managing state:

.. code-block:: python

   from nova.core.memory import Memory

   memory = Memory()
   
   # Store state
   memory.update({
       "current_url": "https://example.com",
       "last_action": "click",
       "timestamp": "2024-04-17T12:00:00Z"
   })
   
   # Retrieve relevant state
   relevant = memory.get_relevant({
       "current_url": "https://example.com"
   })

Tool System
----------

Nova supports a flexible tool system:

.. code-block:: python

   from nova.tools.browser import get_browser_tools
   from nova.core.browser import Browser

   # Create browser tools
   browser = Browser()
   tools = get_browser_tools(browser)

   # Register tools with agent
   agent = TaskAgent(
       llm_config=llm_config,
       memory=memory,
       tools=tools
   )

Error Handling
-------------

Nova includes robust error handling:

.. code-block:: python

   from nova.core.exceptions import NovaError, BrowserError, LLMError

   try:
       result = await agent.run()
   except BrowserError as e:
       print(f"Browser error: {e}")
       # Handle browser-specific errors
   except LLMError as e:
       print(f"LLM error: {e}")
       # Handle LLM-specific errors
   except NovaError as e:
       print(f"General error: {e}")
       # Handle general errors

Advanced Usage
-------------

For more advanced usage, you can subclass the ``BaseAgent`` class:

.. code-block:: python

   from nova.core.base_agent import BaseAgent
   from nova.core.types import AgentState

   class CustomAgent(BaseAgent):
       async def run(self, task: str, task_id: str) -> dict:
           # Custom task execution logic
           self.state = AgentState.RUNNING
           try:
               # Custom implementation
               result = await self._execute_task(task)
               self.state = AgentState.COMPLETED
               return result
           except Exception as e:
               self.state = AgentState.ERROR
               raise

       async def _execute_task(self, task: str) -> dict:
           # Custom task execution logic
           pass

Examples
--------

See the `examples <https://github.com/your-username/nova/tree/main/examples>`_ directory for more examples. 

Using the Tools
--------------

The browser tools are implemented as LangChain BaseTool instances. Here's how to use them:

.. code-block:: python

    # Navigate to a URL
    await tools[0]._arun(url="https://example.com")

    # Click an element
    await tools[1]._arun(selector="#button")

    # Type text
    await tools[2]._arun(selector="#input", text="Hello, World!")

    # Get text content
    text = await tools[3]._arun(selector="#content")

    # Get HTML content
    html = await tools[4]._arun()

    # Take a screenshot
    await tools[5]._arun(path="screenshot.png")

    # Wait for an element
    await tools[6]._arun(selector="#loading", timeout=10)

    # Scroll the page
    await tools[7]._arun(direction="down") 