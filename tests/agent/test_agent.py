import pytest
from typing import Any, Dict, Sequence
from unittest.mock import AsyncMock, MagicMock

from nova.core.agent import Agent
from nova.core.browser import Browser
from nova.core.config import AgentConfig, BrowserConfig
from nova.core.llm import LLM
from nova.core.memory import Memory
from nova.core.tools import Tool


class MockTool(Tool):
    """A mock tool for testing."""

    def __init__(self) -> None:
        super().__init__("mock", "A mock tool")

    async def execute(self, input_data: Dict[str, Any]) -> str:
        return f"Mock result for {input_data}"


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
    tools: Sequence[Tool] = [MockTool()]
    config = AgentConfig()
    browser_config = BrowserConfig(headless=True)
    return Agent(
        llm=mock_llm,
        tools=tools,
        memory=mock_memory,
        config=config,
        browser_config=browser_config,
    )


@pytest.mark.asyncio
async def test_agent_initialization(
    agent: Agent,
    mock_llm: MagicMock,
    mock_browser: MagicMock,
    mock_memory: MagicMock,
) -> None:
    """Test agent initialization."""
    assert agent.llm == mock_llm
    assert len(agent.tools) == 1
    assert agent.memory == mock_memory
    assert isinstance(agent.config, AgentConfig)
    assert isinstance(agent.browser, Browser)


@pytest.mark.asyncio
async def test_agent_run(agent: Agent, mock_llm: MagicMock, mock_browser: MagicMock, mock_memory: MagicMock) -> None:
    """Test running the agent."""
    result = await agent.run("Test task")
    assert result == "Final response"
    
    # Verify LLM was called
    mock_llm.generate_plan.assert_called_once()
    mock_llm.generate_response.assert_called_once()
    
    # Verify memory was used
    mock_memory.get_context.assert_called_once()
    mock_memory.add.assert_called()
    
    # Verify browser was used
    mock_browser.start.assert_called_once()
    mock_browser.stop.assert_called_once()
    mock_browser.navigate.assert_called_once()


@pytest.mark.asyncio
async def test_agent_tool_execution(agent: Agent) -> None:
    """Test tool execution."""
    result = await agent._execute_tool("mock", {"test": "data"})
    assert result == "Mock result for {'test': 'data'}"


@pytest.mark.asyncio
async def test_agent_browser_action(agent: Agent, mock_browser: MagicMock) -> None:
    """Test browser action execution."""
    result = await agent._execute_browser_action({"type": "navigate", "url": "https://example.com"})
    assert result == "Navigated to https://example.com"
    mock_browser.navigate.assert_called_once_with("https://example.com")
