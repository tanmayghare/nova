import pytest
from unittest.mock import AsyncMock, MagicMock

from nova.core.agent import Agent
from nova.core.browser import Browser
from nova.core.config import AgentConfig, BrowserConfig
from nova.core.llm import LLM
from nova.core.memory import Memory
from nova.core.tools import Tool


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM."""
    llm = MagicMock(spec=LLM)
    llm.generate_plan = AsyncMock(return_value=[
        {"type": "tool", "tool": "mock", "input": {"test": "data"}},
        {"type": "browser", "action": {"type": "navigate", "url": "https://example.com"}},
    ])
    llm.generate_response = AsyncMock(return_value="Final response")
    return llm


@pytest.fixture
def mock_browser() -> MagicMock:
    """Create a mock browser."""
    browser = MagicMock(spec=Browser)
    browser.start = AsyncMock()
    browser.stop = AsyncMock()
    browser.navigate = AsyncMock()
    browser.get_text = AsyncMock(return_value="Example text")
    return browser


@pytest.fixture
def mock_memory() -> MagicMock:
    """Create a mock memory."""
    memory = MagicMock(spec=Memory)
    memory.get_context = AsyncMock(return_value="Context")
    memory.add = AsyncMock()
    return memory


@pytest.fixture
def agent(mock_llm: MagicMock, mock_browser: MagicMock, mock_memory: MagicMock) -> Agent:
    """Create an agent instance for testing."""
    tools: list[Tool] = []
    config = AgentConfig()
    browser_config = BrowserConfig(headless=True)
    return Agent(
        llm=mock_llm,
        tools=tools,
        memory=mock_memory,
        config=config,
        browser_config=browser_config,
    )
