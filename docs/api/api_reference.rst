API Reference
============

This document provides a reference for Nova's API.

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

The browser tools are implemented as LangChain BaseTool instances. To get all available tools:

.. autofunction:: nova.tools.browser.get_browser_tools

Individual Tools
~~~~~~~~~~~~~~~

.. autoclass:: nova.tools.browser.NavigateTool
   :members:

.. autoclass:: nova.tools.browser.ClickTool
   :members:

.. autoclass:: nova.tools.browser.TypeTool
   :members:

.. autoclass:: nova.tools.browser.GetTextTool
   :members:

.. autoclass:: nova.tools.browser.GetHtmlTool
   :members:

.. autoclass:: nova.tools.browser.ScreenshotTool
   :members:

.. autoclass:: nova.tools.browser.WaitTool
   :members:

.. autoclass:: nova.tools.browser.ScrollTool
   :members:

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