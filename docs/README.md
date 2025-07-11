# Nova

An intelligent browser automation agent built with Python, powered by LangChain and LangGraph. This project is provided as-is for educational and experimental purposes.

> **⚠️ Notice**: This project is not under active development. If you want to extend or improve the functionality, please fork the repository and create your own version.

## Features

- Browser automation using Playwright
- LLM-powered decision making with LangChain integration
- Flexible LLM provider support (OpenAI, Anthropic, etc.)
- State management and memory system
- Tool-based action execution
- Performance monitoring and logging
- Error handling and recovery

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tanmayghare/nova.git
cd nova
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

4. Install Playwright browser binaries (required for browser automation):
```bash
playwright install  # Installs default browsers like Chromium
# Or install specific ones: playwright install chromium
```

5. Configure environment variables:
   Copy the example environment file and **edit it** with your credentials and preferences (e.g., API keys, model names).
```bash
cp .env.example .env
# --> EDIT .env <--
```

## Usage

The primary way to use Nova is through the `TaskAgent`. Configuration is typically loaded from environment variables defined in your `.env` file.

### Basic Usage (`run_nova.py` / `examples/basic/agent_example.py` style)

This example assumes you have configured your `.env` file.

```python
# Example: examples/basic/agent_example.py
import asyncio
import logging
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file first
load_dotenv()

# Configure basic logging
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentExample")

# Import Nova components after loading .env
from nova.core.llm.llm import LLMConfig
from nova.core.browser import BrowserConfig
from nova.agents.task.task_agent import TaskAgent

async def main():
    task = "Navigate to example.com and find the main heading."
    logger.info(f"Starting Task Agent for task: \"{task}\"")

    agent = None
    try:
        # Initialize configurations (reads from .env)
        llm_config = LLMConfig()
        browser_config = BrowserConfig() # Required for browser tasks

        logger.info(f"LLM Provider: {llm_config.primary_provider}, Model: {llm_config.primary_model}")
        logger.info(f"Browser Headless: {browser_config.headless}")

        # Create the Task Agent
        agent = TaskAgent(
            task_id="example-task-001",
            task_description=task,
            llm_config=llm_config,
            browser_config=browser_config # Automatically enables BrowserTools
            # memory=None, # Uses default Memory if not specified
        )

        # Run the task
        result = await agent.run()

        logger.info("--- Task Execution Result ---")
        print(json.dumps(result.dict(), indent=2))
        logger.info("--- End of Task ---")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
    finally:
        if agent:
            await agent.stop()

if __name__ == "__main__":
    asyncio.run(main())

```

### Advanced Usage (Manual Component Initialization)

While configuration via `.env` is recommended, you can instantiate components manually.

```python
import asyncio
from nova.core.llm.llm import LLMConfig, LLM
from nova.core.browser import BrowserConfig, Browser
from nova.tools.browser import BrowserTools # Corrected import path
from nova.core.memory.memory import MemoryConfig, Memory # Corrected import path
from nova.agents.task.task_agent import TaskAgent
import json

async def main_advanced():
    # Manual LLM configuration
    llm_config = LLMConfig(
        primary_provider="nvidia", # Or "openai", "anthropic", etc.
        primary_model="nvidia/llama-3.3-nemotron-super-49b-v1", # Specify your model
        primary_api_key="YOUR_API_KEY", # Or set NVIDIA_API_KEY env var
        primary_base_url="YOUR_NIM_ENDPOINT", # Or set NVIDIA_NIM_API_BASE_URL env var
        temperature=0.1,
        max_tokens=4096
    )
    # llm = LLM(config=llm_config) # LLM instance is usually created internally by agent

    # Manual browser configuration
    browser_config = BrowserConfig(
        headless=False,
        timeout=30,
        viewport_width=1280,
        viewport_height=720
    )
    # browser = Browser(config=browser_config) # Browser instance is created internally
    # tools = BrowserTools(browser=browser) # Tools are created internally by agent

    # Manual memory configuration
    memory_config = MemoryConfig(max_examples=5)
    memory = Memory(config=memory_config, llm_config=llm_config) # Memory needs LLM config too

    # Create task agent with manual components
    agent = TaskAgent(
        task_id="advanced-example-001",
        task_description="Go to duckduckgo.com, search for 'asyncio python', and list the first 3 results.",
        llm_config=llm_config,
        browser_config=browser_config, # Pass config, not instance
        memory=memory # Pass memory instance
        # tools=[tools] # Explicit tool passing is possible but often not needed
    )

    try:
        result = await agent.run()
        print(json.dumps(result.dict(), indent=2))
    finally:
        await agent.stop()


if __name__ == "__main__":
    # asyncio.run(main()) # Basic example
    asyncio.run(main_advanced()) # Advanced example

```

## Project Structure

```
nova/
├── .env.example        # Example environment file
├── run_nova.py         # Main execution script (basic example)
├── requirements.txt    # Core dependencies
├── setup.py            # Package setup
├── pyproject.toml      # Build system and tool config (ruff, pytest)
├── src/
│   └── nova/
│       ├── agents/           # Agent implementations (TaskAgent, etc.)
│       │   └── task/
│       ├── core/             # Core components
│       │   ├── browser/      # Browser control (Browser, BrowserConfig)
│       │   ├── config/       # Core configuration (schemas other than LLM/Browser)
│       │   │   └── config.py
│       │   ├── llm/          # LLM abstraction (LLM, LLMConfig)
│       │   │   └── llm.py
│       │   ├── memory/       # Memory system (Memory, MemoryConfig)
│       │   │   └── memory.py
│       │   └── tools/        # Tool registry, base tool, result (ToolRegistry, Tool, ToolResult)
│       │       ├── tool.py
│       │       └── tools.py
│       ├── tools/            # Tool implementations (distinct from core/tools)
│       │   └── browser/      # Browser-specific tools (BrowserTools, schemas)
│       │       └── browser_tools.py
│       └── langchain/        # LangChain/LangGraph specific integrations (if any)
├── tests/                    # Unit and integration tests
├── docs/                     # Documentation files (like this one)
├── examples/                 # Usage examples (basic, advanced)
│   └── basic/
│       └── agent_example.py  # Example matching run_nova.py
└── logs/                     # Default log output directory (if configured)
```

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
pytest
```

3. Run linting:
```bash
make lint
```

4. Run type checking:
```bash
make typecheck
```

## Configuration

Nova configuration is primarily managed through environment variables loaded from an `.env` file. See `.env.example` for a full list. Key variables include:

- **LLM Configuration:**
  - `LLM_PROVIDER`: `nvidia`, `openai`, `anthropic`, etc. (Default: `nvidia`)
  - `MODEL_NAME`: Specific model identifier (e.g., `nvidia/llama-3.3-nemotron-super-49b-v1`, `gpt-4-turbo`)
  - `NVIDIA_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`: API key for the chosen provider.
  - `NVIDIA_API_BASE_URL`: Endpoint for NVIDIA provider (if not default).
  - `MODEL_TEMPERATURE`: (e.g., `0.1`)
  - `MODEL_MAX_TOKENS`: (e.g., `4096`)
  - `MODEL_TOP_P`, `MODEL_TOP_K`, `MODEL_REPETITION_PENALTY`, etc.
- **Browser Configuration:**
  - `BROWSER_HEADLESS`: `true` or `false`.
  - `BROWSER_TIMEOUT`: Default timeout for browser actions (seconds).
  - `BROWSER_VIEWPORT_WIDTH`, `BROWSER_VIEWPORT_HEIGHT`: Browser window size.
- **Memory Configuration:**
  - `MEMORY_MAX_EXAMPLES`: Number of examples to store in memory.
- **Logging:**
  - `LOG_LEVEL`: `INFO`, `DEBUG`, `WARNING`, `ERROR`.

See the configuration classes for more details:
- `src/nova/core/llm/llm.py` (`LLMConfig`)
- `src/nova/core/browser/config.py` (`BrowserConfig`)
- `src/nova/core/memory/memory.py` (`MemoryConfig`)

## License

MIT License 