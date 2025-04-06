Welcome to Nova's documentation!
================================

Nova is an intelligent browser automation agent built with Python. It uses LLMs to make decisions and Playwright for browser automation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api
   contributing

Features
--------

* Browser automation using Playwright
* LLM-powered decision making
* State management
* Memory system for short-term and long-term state
* Action execution system
* Error handling

Installation
-----------

.. code-block:: bash

   pip install nova

Quick Start
----------

.. code-block:: python

   from nova import Agent
   from langchain_openai import ChatOpenAI
   import os
   from dotenv import load_dotenv

   load_dotenv()

   llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
   agent = Agent(
       task="Navigate to example.com and click the first link",
       llm=llm
   )
   result = await agent.run()
   print(result)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 