Installation
============

Prerequisites
------------

Nova requires Python 3.9 or higher and Node.js for Playwright support. It supports multiple LLM providers including OpenAI and Anthropic.

Installing Python
----------------

If you don't have Python installed, you can download it from the `official Python website <https://www.python.org/downloads/>`_.

Installing Node.js
-----------------

Playwright requires Node.js. You can install it from the `official Node.js website <https://nodejs.org/>`_.

Installing Nova
--------------

You can install Nova using pip:

.. code-block:: bash

   pip install nova

For development, you can install from source:

.. code-block:: bash

   git clone https://github.com/your-username/nova.git
   cd nova
   pip install -e ".[dev]"

Installing Dependencies
----------------------

After installing Nova, you need to install the Playwright browsers:

.. code-block:: bash

   playwright install chromium

Environment Variables
-------------------

Create a ``.env`` file in your project root with the following variables:

.. code-block:: text

   # LLM Configuration
   LLM_PROVIDER=openai  # or anthropic
   LLM_API_KEY=your-api-key
   LLM_MODEL=gpt-3.5-turbo  # or claude-3-opus-20240229
   LLM_TEMPERATURE=0.1
   LLM_MAX_TOKENS=1000

   # Browser Configuration
   BROWSER_HEADLESS=true
   BROWSER_TIMEOUT=30
   BROWSER_VIEWPORT_WIDTH=1280
   BROWSER_VIEWPORT_HEIGHT=720

   # Memory Configuration
   MEMORY_MAX_ENTRIES=1000
   MEMORY_CLEANUP_INTERVAL=3600

   # Logging Configuration
   LOG_LEVEL=INFO
   LOG_DIR=logs
   LOG_FORMAT=json

Configuration
------------

Nova can be configured through environment variables or programmatically:

.. code-block:: python

   from nova.core.config import LLMConfig, AgentConfig
   from nova.core.browser import Browser
   from nova.core.memory import Memory

   # LLM Configuration
   llm_config = LLMConfig(
       provider="openai",
       model_name="gpt-3.5-turbo",
       temperature=0.1,
       max_tokens=1000
   )

   # Agent Configuration
   config = AgentConfig(
       llm_config=llm_config,
       max_steps=50,
       timeout=300,
       verbose=True
   )

   # Browser Configuration
   browser = Browser(
       headless=False,
       timeout=30,
       viewport={"width": 1280, "height": 720}
   )

   # Memory Configuration
   memory = Memory()

Verifying Installation
---------------------

You can verify your installation by running:

.. code-block:: bash

   python -c "import nova; print(nova.__version__)"

This should print the version number of Nova.

Troubleshooting
--------------

If you encounter any issues during installation, please check the following:

1. Make sure you have Python 3.9 or higher installed
2. Verify that Node.js is installed and in your PATH
3. Ensure Playwright browsers are installed
4. Check that your LLM provider credentials are correctly configured
5. Ensure all dependencies are installed correctly

Common Issues
------------

1. **Browser Installation Issues**:
   - Run ``playwright install --force`` to reinstall browsers
   - Check system dependencies for Playwright

2. **LLM Connection Issues**:
   - Verify API keys are correct
   - Check network connectivity
   - Ensure model names are correct

3. **Memory Issues**:
   - Check available system memory
   - Adjust ``MEMORY_MAX_ENTRIES`` if needed
   - Monitor memory usage during execution

4. **Performance Issues**:
   - Adjust browser timeout settings
   - Optimize LLM parameters
   - Monitor system resources

If you still have issues, please open an issue on the GitHub repository.