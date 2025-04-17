Usage
=====

Basic Usage
----------

Here's a basic example of how to use Nova:

.. code-block:: python

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
   memory = Memory()
   config = AgentConfig(llm_config=llm_config)

   # Create task agent
   agent = TaskAgent(
       llm_config=llm_config,
       memory=memory,
       tools=None  # Agent will register browser tools automatically
   )

   # Run a task
   result = await agent.run(
       task="Navigate to example.com and click the first link",
       task_id="test-001"
   )
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
   from nova.tools.browser_tools import BrowserTools

   # Custom browser configuration
   browser = Browser(
       headless=False,
       timeout=30,
       viewport={"width": 1280, "height": 720}
   )

   # Create browser tools
   tools = BrowserTools(browser)

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

   from nova.tools.browser_tools import BrowserTools
   from nova.core.browser import Browser

   # Create browser tools
   browser = Browser()
   tools = BrowserTools(browser)

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
       result = await agent.run(
           task="Navigate to example.com",
           task_id="test-001"
       )
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