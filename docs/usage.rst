Usage
=====

Basic Usage
----------

Here's a basic example of how to use Nova:

.. code-block:: python

   from nova import Agent
   from nova.core.llama import LlamaModel
   import os
   from dotenv import load_dotenv

   load_dotenv()

   # Initialize the language model
   model = LlamaModel(model_name="llama3.2:3b-instruct-q8_0")

   # Create an agent
   agent = Agent(
       task="Navigate to example.com and click the first link",
       llm=model
   )

   # Run the agent
   result = await agent.run()
   print(result)

Configuration
------------

You can configure the agent using the ``AgentConfig`` class:

.. code-block:: python

   from nova import Agent, AgentConfig

   config = AgentConfig(
       max_steps=50,
       timeout=300,
       verbose=True
   )

   agent = Agent(
       task="Your task here",
       llm=model,
       config=config
   )

Browser Configuration
-------------------

You can configure the browser using the ``BrowserConfig`` class:

.. code-block:: python

   from nova import BrowserConfig

   browser_config = BrowserConfig(
       headless=True,
       viewport_width=1280,
       viewport_height=720
   )

   agent = Agent(
       task="Your task here",
       llm=model,
       browser_config=browser_config
   )

Memory System
------------

Nova includes a memory system for managing state:

.. code-block:: python

   from nova import Memory

   memory = Memory()
   memory.update({"url": "https://example.com"})
   relevant = memory.get_relevant({"url": "https://example.com"})

Actions
-------

Nova supports various browser actions:

.. code-block:: python

   from nova.types.actions import Action

   # Navigation
   action = Action(type="navigate", parameters={"url": "https://example.com"})

   # Click
   action = Action(type="click", parameters={"selector": "button"})

   # Type
   action = Action(type="type", parameters={"selector": "input", "text": "Hello"})

Error Handling
-------------

Nova includes robust error handling:

.. code-block:: python

   try:
       result = await agent.run()
   except Exception as e:
       print(f"Error: {e}")
       # Handle error

Advanced Usage
-------------

For more advanced usage, you can subclass the ``Agent`` class:

.. code-block:: python

   from nova import Agent

   class CustomAgent(Agent):
       async def _get_next_action(self, state):
           # Custom logic for determining next action
           pass

       async def _execute_action(self, action):
           # Custom logic for executing actions
           pass

Examples
--------

See the `examples <https://github.com/your-username/nova/tree/main/examples>`_ directory for more examples. 