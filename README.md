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

## Usage

```python
from nova.core.agent import Agent
from langchain_openai import ChatOpenAI

# Initialize the agent
agent = Agent(
    task="Navigate to example.com and click the first link",
    llm=ChatOpenAI(model="gpt-4"),
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