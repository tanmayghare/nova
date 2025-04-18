import pytest
import json
import os
from typing import List

from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from nova.core.llm import LangChainAdapter, LLM, LLMConfig


@pytest.fixture
def mock_model():
    """Create a mock LangChain model."""
    model = MagicMock(spec=BaseChatModel)
    model.ainvoke = AsyncMock()
    return model


@pytest.fixture
def llm_config_defaults():
    # Define default parameters used when no env vars are set
    # Align these with LLMConfig defaults
    return {
        "primary_provider": "nvidia",
        "primary_model": "meta/llama-3.3-70b-instruct",
        "primary_base_url": None,
        "primary_api_key": None,
        "temperature": 0.1,
        "max_tokens": 4096,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "streaming": True,
        "timeout": 15.0,
        "max_retries": 3,
        "retry_delay": 1.0,
        "batch_size": 4
    }


@pytest.fixture
def mock_chat_model():
    model = AsyncMock(spec=BaseChatModel)
    model.invoke.return_value = AIMessage(content="Test response")
    return model


@pytest.fixture
def llm(llm_config_defaults):
    # Use defaults for the basic LLM fixture if needed for other tests
    # Note: Most LLMConfig tests will create their own instance
    config = LLMConfig(**llm_config_defaults)
    return LLM(config)


@pytest.mark.asyncio
async def test_generate_plan_success(mock_model):
    """Test successful plan generation."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.content = """[
        {"type": "browser", "action": {"type": "navigate", "url": "https://example.com"}},
        {"type": "browser", "action": {"type": "click", "selector": "#submit"}},
        {"type": "tool", "tool": "screenshot", "input": {"path": "result.png"}}
    ]"""
    mock_model.ainvoke.return_value = mock_response
    
    adapter = LangChainAdapter(mock_model)
    task = "Navigate to example.com and take a screenshot"
    context = "Previous attempts failed"
    
    plan = await adapter.generate_plan(task, context)
    
    assert len(plan) == 3
    assert plan[0]["type"] == "browser"
    assert plan[0]["action"]["type"] == "navigate"
    assert plan[2]["type"] == "tool"
    assert plan[2]["tool"] == "screenshot"


@pytest.mark.asyncio
async def test_generate_plan_invalid_json(mock_model):
    """Test plan generation with invalid JSON response."""
    mock_response = MagicMock()
    mock_response.content = "Invalid JSON response"
    mock_model.ainvoke.return_value = mock_response
    
    adapter = LangChainAdapter(mock_model)
    
    with pytest.raises(RuntimeError, match="Failed to generate valid plan"):
        await adapter.generate_plan("test task", "context")


@pytest.mark.asyncio
async def test_generate_plan_invalid_steps(mock_model):
    """Test plan generation with invalid step format."""
    mock_response = MagicMock()
    mock_response.content = '[{"invalid": "step"}]'
    mock_model.ainvoke.return_value = mock_response
    
    adapter = LangChainAdapter(mock_model)
    
    with pytest.raises(RuntimeError, match="Failed to generate valid plan"):
        await adapter.generate_plan("test task", "context")


@pytest.mark.asyncio
async def test_generate_response_success(mock_model):
    """Test successful response generation."""
    mock_response = MagicMock()
    mock_response.content = "Task completed successfully"
    mock_model.ainvoke.return_value = mock_response
    
    adapter = LangChainAdapter(mock_model)
    task = "test task"
    plan = [
        {"type": "browser", "action": {"type": "navigate", "url": "https://example.com"}}
    ]
    context = "Previous context"
    
    response = await adapter.generate_response(task, plan, context)
    
    assert response == "Task completed successfully"
    mock_model.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_generate_response_error(mock_model):
    """Test response generation with error."""
    mock_model.ainvoke.side_effect = Exception("Model error")
    
    adapter = LangChainAdapter(mock_model)
    
    with pytest.raises(Exception):
        await adapter.generate_response("test task", [], "context")


# ---------------------- Additional Tests ----------------------

@pytest.mark.asyncio
async def test_generate_plan_with_text_wrapping(mock_model):
    """Test plan generation when JSON is wrapped in text."""
    mock_response = MagicMock()
    mock_response.content = """Here's the plan:
    [
        {"type": "browser", "action": {"type": "navigate", "url": "https://example.com"}}
    ]
    Hope this helps!"""
    mock_model.ainvoke.return_value = mock_response
    
    adapter = LangChainAdapter(mock_model)
    
    plan = await adapter.generate_plan("test task", "context")
    assert len(plan) == 1
    assert plan[0]["type"] == "browser"


@pytest.mark.asyncio
async def test_generate_plan_empty_response(mock_model):
    """Test plan generation with empty response."""
    mock_response = MagicMock()
    mock_response.content = ""
    mock_model.ainvoke.return_value = mock_response
    
    adapter = LangChainAdapter(mock_model)
    
    with pytest.raises(RuntimeError, match="Failed to generate valid plan"):
        await adapter.generate_plan("test task", "context")


@pytest.mark.asyncio
async def test_generate_plan_different_response_format(mock_model):
    """Test handling different response object formats."""
    # Test string response instead of object with content attribute
    mock_model.ainvoke.return_value = """[
        {"type": "browser", "action": {"type": "navigate", "url": "https://example.com"}}
    ]"""
    
    adapter = LangChainAdapter(mock_model)
    plan = await adapter.generate_plan("test task", "context")
    
    assert len(plan) == 1
    assert plan[0]["type"] == "browser"


@pytest.mark.asyncio
async def test_generate_plan_complex_json(mock_model):
    """Test with complex nested JSON structures."""
    complex_json = [
        {
            "type": "browser", 
            "action": {
                "type": "fill_form", 
                "fields": [
                    {"selector": "#name", "value": "Test User"},
                    {"selector": "#email", "value": "test@example.com"}
                ],
                "submit": True
            }
        },
        {
            "type": "tool",
            "tool": "api_call",
            "input": {
                "url": "https://api.example.com",
                "method": "POST",
                "headers": {"Content-Type": "application/json"},
                "data": {"key": "value"}
            }
        }
    ]
    
    mock_response = MagicMock()
    mock_response.content = json.dumps(complex_json)
    mock_model.ainvoke.return_value = mock_response
    
    adapter = LangChainAdapter(mock_model)
    plan = await adapter.generate_plan("test task", "context")
    
    assert len(plan) == 2
    assert plan[0]["action"]["fields"][0]["selector"] == "#name"
    assert plan[1]["input"]["headers"]["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_generate_plan_invalid_browser_action(mock_model):
    """Test with invalid browser action format."""
    mock_response = MagicMock()
    mock_response.content = """[
        {"type": "browser", "not_action": "invalid"}
    ]"""
    mock_model.ainvoke.return_value = mock_response
    
    adapter = LangChainAdapter(mock_model)
    
    with pytest.raises(RuntimeError, match="Failed to generate valid plan"):
        await adapter.generate_plan("test task", "context")


@pytest.mark.asyncio
async def test_generate_plan_invalid_tool_action(mock_model):
    """Test with invalid tool action format."""
    mock_response = MagicMock()
    mock_response.content = """[
        {"type": "tool", "tool": "screenshot"}
    ]"""
    mock_model.ainvoke.return_value = mock_response
    
    adapter = LangChainAdapter(mock_model)
    
    with pytest.raises(RuntimeError, match="Failed to generate valid plan"):
        await adapter.generate_plan("test task", "context")


@pytest.mark.asyncio
async def test_generate_response_different_format(mock_model):
    """Test response generation with different response formats."""
    # Test string response
    mock_model.ainvoke.return_value = "Direct string response"
    
    adapter = LangChainAdapter(mock_model)
    response = await adapter.generate_response("test", [], "context")
    
    assert response == "Direct string response"


@pytest.mark.asyncio
async def test_generate_plan_model_timeout_simulation():
    """Test handling of model timeouts."""
    model = MagicMock(spec=BaseChatModel)
    model.ainvoke = AsyncMock(side_effect=TimeoutError("Model timeout"))
    
    adapter = LangChainAdapter(model)
    
    with pytest.raises(Exception):
        await adapter.generate_plan("test task", "context")


@pytest.mark.asyncio
async def test_llm_initialization(llm, llm_config_defaults):
    """Test that the LLM wrapper initializes correctly."""
    assert llm.config == llm_config_defaults
    assert llm.config.primary_provider == "nvidia"
    assert llm.config.primary_model == "meta/llama3-70b-instruct"


@patch('os.getenv')
def test_llm_config_defaults(mock_getenv, llm_config_defaults):
    """Test LLMConfig initializes with defaults when no env vars are set."""
    # Ensure os.getenv returns None for all queried vars to force defaults
    mock_getenv.return_value = None

    config = LLMConfig()

    # Assert against the defined defaults
    assert config.primary_provider == llm_config_defaults["primary_provider"]
    assert config.primary_model == llm_config_defaults["primary_model"]
    assert config.primary_base_url == llm_config_defaults["primary_base_url"] # Should be None
    assert config.temperature == llm_config_defaults["temperature"]
    assert config.max_tokens == llm_config_defaults["max_tokens"]
    assert config.top_p == llm_config_defaults["top_p"]
    assert config.top_k == llm_config_defaults["top_k"]
    assert config.repetition_penalty == llm_config_defaults["repetition_penalty"]
    assert config.streaming == llm_config_defaults["streaming"]
    assert config.timeout == llm_config_defaults["timeout"]
    assert config.max_retries == llm_config_defaults["max_retries"]
    assert config.retry_delay == llm_config_defaults["retry_delay"]
    assert config.batch_size == llm_config_defaults["batch_size"]
    # API key should also be None if not set
    assert config.primary_api_key is None


@patch.dict(os.environ, {
    "LLM_PROVIDER": "nvidia",
    "MODEL_NAME": "nvidia/nemotron-test",
    "NVIDIA_API_KEY": "nvapi-testkey123",
    "NVIDIA_API_BASE_URL": "https://api.example.com/v1",
})
def test_llm_config_nvidia_provider():
    """Test LLMConfig specific handling for NVIDIA provider."""
    config = LLMConfig()

    assert config.primary_provider == "nvidia"
    assert config.primary_model == "nvidia/nemotron-test"
    assert config.primary_api_key == "nvapi-testkey123"
    assert config.primary_base_url == "https://api.example.com/v1"


@patch.dict(os.environ, {
    "LLM_PROVIDER": "nvidia",
    "MODEL_NAME": "nvidia/nemotron-test",
    # Missing NVIDIA_API_KEY
})
def test_llm_config_nvidia_missing_key_validation():
    """Test validation fails if NVIDIA provider is used without API key."""
    with pytest.raises(ValueError, match="NVIDIA_API_KEY environment variable must be set"):
        LLMConfig()

@patch.dict(os.environ, {"LLM_PROVIDER": "nvidia", "NVIDIA_API_KEY": "dummykey"}) # Minimal valid NVIDIA
def test_llm_config_nvidia_default_model_and_url():
    """Test default model and base URL for NVIDIA if not specified."""
    config = LLMConfig()
    # Check against defaults defined in LLMConfig
    assert config.primary_provider == "nvidia"
    assert config.primary_model == "meta/llama3-70b-instruct" # Default NVIDIA model
    assert config.primary_base_url is None # Default NVIDIA URL is None


@patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "ak-key123"})
def test_llm_config_other_providers():
    """Test config loading for a provider other than NIM/OpenAI."""
    config = LLMConfig()
    assert config.primary_provider == "anthropic"
    assert config.primary_api_key == "ak-key123"
    # Check that a default model is assigned if MODEL_NAME is not set
    assert config.primary_model == "meta/llama3-70b-instruct" # Should fall back to global default


@patch.dict(os.environ, {"MODEL_TEMPERATURE": "2.5"})
def test_llm_config_invalid_temperature():
    """Test validation for temperature out of range."""
    with pytest.raises(ValueError, match="Temperature must be between 0.0 and 2.0"):
        LLMConfig()

@patch.dict(os.environ, {"MODEL_MAX_TOKENS": "0"})
def test_llm_config_invalid_max_tokens():
    """Test validation for max_tokens out of range."""
    with pytest.raises(ValueError, match="Max tokens must be positive"):
        LLMConfig()


@patch('nova.core.llm.llm.LLM._init_llm') # Patch internal init method
def test_llm_creation_triggers_init(mock_init, llm_config_defaults):
    """Test that creating an LLM instance calls its internal _init_llm."""
    config = LLMConfig(**llm_config_defaults)
    llm_instance = LLM(config=config)
    mock_init.assert_called_once()


@patch('nova.core.llm.ChatOpenAI')
def test_llm_generate_text(mock_chat, llm):
    mock_chat.return_value.invoke.return_value.content = "Test response"
    response = llm.generate_text("Test prompt")
    assert response == "Test response"
    mock_chat.return_value.invoke.assert_called_once()


@patch('nova.core.llm.ChatOpenAI')
def test_llm_generate_plan(mock_chat, llm):
    mock_chat.return_value.invoke.return_value.content = '{"steps": [{"action": "test"}]}'
    plan = llm.generate_plan("Test task")
    assert isinstance(plan, dict)
    assert "steps" in plan
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["action"] == "test"


@patch('nova.core.llm.ChatOpenAI')
def test_llm_interpret_command(mock_chat, llm):
    mock_chat.return_value.invoke.return_value.content = '{"command": "test"}'
    interpretation = llm.interpret_command("Test command")
    assert isinstance(interpretation, dict)
    assert "command" in interpretation
    assert interpretation["command"] == "test"


@patch('nova.core.llm.ChatOpenAI')
def test_llm_generate_recovery_plan(mock_chat, llm):
    mock_chat.return_value.invoke.return_value.content = '{"recovery_steps": ["step1"]}'
    recovery_plan = llm.generate_recovery_plan("Test error")
    assert isinstance(recovery_plan, dict)
    assert "recovery_steps" in recovery_plan
    assert len(recovery_plan["recovery_steps"]) == 1
    assert recovery_plan["recovery_steps"][0] == "step1"


@pytest.mark.asyncio
async def test_batch_processing():
    """Test batch processing of multiple prompts."""
    llm = LLM(batch_size=2)
    
    # Test with multiple prompts
    prompts = [
        "What is 2+2?",
        "What is 3+3?",
        "What is 4+4?",
        "What is 5+5?"
    ]
    
    responses = await llm.generate_batch(prompts)
    assert len(responses) == len(prompts)
    for response in responses:
        assert isinstance(response, str)
        assert len(response) > 0


@pytest.mark.asyncio
async def test_streaming():
    """Test streaming response generation."""
    llm = LLM(enable_streaming=True)
    
    # Test streaming response
    prompt = "Count from 1 to 5"
    chunks: List[str] = []
    
    async for chunk in llm.generate_stream(prompt):
        chunks.append(chunk)
    
    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert len(full_response) > 0


@pytest.mark.asyncio
async def test_token_counting():
    """Test token counting functionality."""
    llm = LLM()
    
    # Initial count should be 0
    assert llm.get_token_count() == 0
    
    # Generate some responses
    await llm.generate("Test prompt 1")
    count1 = llm.get_token_count()
    assert count1 > 0
    
    await llm.generate("Test prompt 2")
    count2 = llm.get_token_count()
    assert count2 > count1
    
    # Test reset
    llm.reset_token_count()
    assert llm.get_token_count() == 0


@pytest.mark.asyncio
async def test_streaming_disabled():
    """Test behavior when streaming is disabled."""
    llm = LLM(enable_streaming=False)
    
    # Test streaming response (should return full response)
    prompt = "Test prompt"
    chunks: List[str] = []
    
    async for chunk in llm.generate_stream(prompt):
        chunks.append(chunk)
    
    assert len(chunks) == 1
    assert len(chunks[0]) > 0


@pytest.mark.asyncio
async def test_chat():
    """Test chat functionality."""
    llm = LLM()
    
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"}
    ]
    
    response = await llm.chat(messages)
    assert isinstance(response, str)
    assert len(response) > 0 