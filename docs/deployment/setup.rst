Installation
============

Prerequisites
------------

Nova requires Python 3.9 or higher. It also requires Node.js for Playwright and Ollama for local LLM support.

Installing Python
----------------

If you don't have Python installed, you can download it from the `official Python website <https://www.python.org/downloads/>`_.

Installing Node.js
-----------------

Playwright requires Node.js. You can install it from the `official Node.js website <https://nodejs.org/>`_.

Installing Ollama
----------------

Ollama is required for local LLM support. You can install it from the `official Ollama website <https://ollama.ai/>`_.

After installation, pull the required model:

.. code-block:: bash

   ollama pull mistral-small3.1:24b-instruct-2503-q4_K_M

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

   # Browser configuration
   BROWSER_HEADLESS=true
   BROWSER_VIEWPORT_WIDTH=1280
   BROWSER_VIEWPORT_HEIGHT=720

   # Optional Llama configuration
   LLAMA_MODEL_PATH=~/<path-to-your-llama-model.gguf>
   LLAMA_N_CTX=4096
   LLAMA_N_THREADS=6
   LLAMA_N_GPU_LAYERS=1

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
3. Ensure Ollama is installed and running
4. Check that your Llama model is properly configured
5. Ensure all dependencies are installed correctly

If you still have issues, please open an issue on the GitHub repository.