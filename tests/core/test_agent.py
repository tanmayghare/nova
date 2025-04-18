"""Tests for agent functionality."""

import pytest

from nova.core.agent.langchain_agent import LangChainAgent
from nova.core.llm import LLMConfig
from nova.tools.browser import get_browser_tools

@pytest.fixture
def mock_llm():
    from tests.core.mock_llm import MockLLM
    return MockLLM()

@pytest.fixture
def mock_browser():
    from tests.core.mock_browser import MockBrowserTool
    return MockBrowserTool()

@pytest.fixture
def mock_memory():
    from tests.core.mock_memory import MockMemory
    return MockMemory()

@pytest.fixture
def mock_monitor():
    from tests.core.mock_monitor import MockMonitor
    return MockMonitor()

@pytest.fixture
def agent(mock_llm, mock_browser, mock_memory, mock_monitor):
    config = LLMConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=1000
    )
    return LangChainAgent(
        llm=mock_llm,
        browser=mock_browser,
        memory=mock_memory,
        monitor=mock_monitor
    )

def test_agent_initialization(agent):
    assert agent.llm is not None
    assert agent.browser is not None
    assert agent.memory is not None
    assert agent.monitor is not None

@pytest.mark.asyncio
async def test_agent_run(agent):
    result = await agent.run("Test task")
    assert result["status"] == "success"
    assert result["output"] is not None

@pytest.mark.asyncio
async def test_agent_cleanup(agent):
    await agent.cleanup()
    assert agent.monitor.get_metrics()["total_tasks"] > 0

@pytest.mark.asyncio
async def test_agent_tool_execution(agent):
    tools = get_browser_tools(agent.browser)
    agent.tools = tools
    result = await agent.run("Navigate to example.com")
    assert result["status"] == "success"
    assert agent.monitor.get_metrics()["tool_calls"] > 0

@pytest.mark.asyncio
async def test_agent_performance_metrics(agent):
    await agent.run("Test task")
    metrics = agent.monitor.get_metrics()
    assert metrics["llm_calls"] > 0
    assert metrics["tool_calls"] > 0
    assert metrics["memory_operations"] > 0
    assert metrics["chain_steps"] > 0 