import pytest
from unittest.mock import Mock, patch
from nova.core.base_agent import BaseAgent, AgentState
from nova.core.llm import LLMConfig
from nova.core.memory import Memory
from nova.tools.browser_tools import BrowserTools

@pytest.fixture
def mock_llm():
    return Mock()

@pytest.fixture
def mock_browser():
    return Mock()

@pytest.fixture
def mock_memory():
    return Mock(spec=Memory)

@pytest.fixture
def agent(mock_llm, mock_browser, mock_memory):
    config = LLMConfig(
        provider="openai",
        model_name="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=1000
    )
    return BaseAgent(
        llm_config=config,
        browser=mock_browser,
        memory=mock_memory
    )

def test_agent_initialization(agent):
    assert agent.state == AgentState.IDLE
    assert agent.llm_config.provider == "openai"
    assert agent.llm_config.model_name == "gpt-3.5-turbo"

def test_agent_start(agent):
    agent.start()
    assert agent.state == AgentState.RUNNING

def test_agent_stop(agent):
    agent.start()
    agent.stop()
    assert agent.state == AgentState.IDLE

def test_agent_cleanup(agent):
    agent.start()
    agent.cleanup()
    assert agent.state == AgentState.IDLE
    agent.browser.quit.assert_called_once()

def test_agent_register_tools(agent):
    tools = BrowserTools(agent.browser)
    agent.register_tools(tools)
    assert len(agent.tools) > 0

def test_agent_performance_metrics(agent):
    agent.start()
    agent.stop()
    assert agent.performance_metrics["total_runtime"] > 0
    assert agent.performance_metrics["total_tasks"] == 0 