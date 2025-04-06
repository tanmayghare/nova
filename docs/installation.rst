Installation
============

Prerequisites
------------

Nova requires Python 3.9 or higher. It also requires Node.js for Playwright.

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
   pip install -e .

Installing Dependencies
----------------------

After installing Nova, you need to install the Playwright browsers:

.. code-block:: bash

   playwright install

Environment Variables
-------------------

Create a ``.env`` file in your project root with the following variables:

.. code-block:: text

   # Required
   OPENAI_API_KEY=your_openai_api_key

   # Optional
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GOOGLE_API_KEY=your_google_api_key

   # Browser configuration
   BROWSER_HEADLESS=true
   BROWSER_VIEWPORT_WIDTH=1280
   BROWSER_VIEWPORT_HEIGHT=720

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
3. Check that your API keys are correctly set in the .env file
4. Ensure all dependencies are installed correctly

If you still have issues, please open an issue on the GitHub repository. 