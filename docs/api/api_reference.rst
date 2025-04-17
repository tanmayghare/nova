API Reference
============

Core Classes
-----------

BaseAgent
~~~~~~~~~

.. autoclass:: nova.core.base_agent.BaseAgent
   :members:
   :undoc-members:
   :show-inheritance:

TaskAgent
~~~~~~~~~

.. autoclass:: nova.agents.task.task_agent.TaskAgent
   :members:
   :undoc-members:
   :show-inheritance:

LLM
~~~

.. autoclass:: nova.core.llm.LLM
   :members:
   :undoc-members:
   :show-inheritance:

LLMConfig
~~~~~~~~~

.. autoclass:: nova.core.config.LLMConfig
   :members:
   :undoc-members:
   :show-inheritance:

Browser
~~~~~~~

.. autoclass:: nova.core.browser.Browser
   :members:
   :undoc-members:
   :show-inheritance:

BrowserTools
~~~~~~~~~~~

.. autoclass:: nova.tools.browser_tools.BrowserTools
   :members:
   :undoc-members:
   :show-inheritance:

Memory
~~~~~~

.. autoclass:: nova.core.memory.Memory
   :members:
   :undoc-members:
   :show-inheritance:

InteractionLogger
~~~~~~~~~~~~~~~

.. autoclass:: nova.core.logging.InteractionLogger
   :members:
   :undoc-members:
   :show-inheritance:

Types
-----

AgentState
~~~~~~~~~

.. autoclass:: nova.core.base_agent.AgentState
   :members:
   :undoc-members:
   :show-inheritance:

Action
~~~~~~

.. autoclass:: nova.core.types.Action
   :members:
   :undoc-members:
   :show-inheritance:

ActionResult
~~~~~~~~~~~

.. autoclass:: nova.core.types.ActionResult
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
---------

NovaError
~~~~~~~~~

.. autoexception:: nova.core.exceptions.NovaError
   :members:
   :undoc-members:
   :show-inheritance:

BrowserError
~~~~~~~~~~~

.. autoexception:: nova.core.exceptions.BrowserError
   :members:
   :undoc-members:
   :show-inheritance:

ActionError
~~~~~~~~~~

.. autoexception:: nova.core.exceptions.ActionError
   :members:
   :undoc-members:
   :show-inheritance:

LLMError
~~~~~~~~

.. autoexception:: nova.core.exceptions.LLMError
   :members:
   :undoc-members:
   :show-inheritance: 