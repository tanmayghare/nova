# Nova

An intelligent browser automation agent built with Python.

## Features

- Browser automation using Playwright
- LLM-powered decision making
- State management
- Memory system
- Action execution
- Error handling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nova.git
cd nova
```

2. Install dependencies:
```bash
pip install -e .
```

3. Install Playwright:
```bash
playwright install chromium
```

4. Install Ollama and pull the model:
```bash
ollama pull mistral-small3.1:24b-instruct-2503-q4_K_M
```

## Usage

```python
from nova.core.agent import Agent
from nova.core.llama import LlamaModel

# Initialize the agent
model = LlamaModel(model_name="mistral-small3.1:24b-instruct-2503-q4_K_M")
agent = Agent(
    task="Navigate to example.com and click the first link",
    llm=model
)

# Run the agent
result = await agent.run()
print(result)
```

## Project Structure

```
nova/
├── core/           # Core components
│   ├── agent.py    # Main agent class
│   ├── browser.py  # Browser control
│   ├── memory.py   # Memory system
│   └── state.py    # State management
├── types/          # Type definitions
│   └── actions.py  # Action types
├── tests/          # Test suite
└── examples/       # Usage examples
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

## License

MIT License 