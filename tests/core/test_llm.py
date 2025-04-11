import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.language_models.chat_models import BaseChatModel
from nova.core.llm import LangChainAdapter, LLM
from typing import List


@pytest.fixture
def mock_model():
    """Create a mock LangChain model."""
    model = MagicMock(spec=BaseChatModel)
    model.ainvoke = AsyncMock()
    return model


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
async def test_llm_initialization():
    """Test that the LLM initializes correctly."""
    llm = LLM()
    assert llm is not None


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