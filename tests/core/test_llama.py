"""Tests for Llama model integration."""

import pytest
import asyncio
from typing import List
from nova.core.llama import LlamaModel

@pytest.mark.asyncio
async def test_llama_model_initialization():
    """Test that the Llama model initializes correctly."""
    model = LlamaModel()
    assert model is not None

@pytest.mark.asyncio
async def test_batch_processing():
    """Test batch processing of multiple prompts."""
    model = LlamaModel(batch_size=2)
    
    # Test with multiple prompts
    prompts = [
        "What is 2+2?",
        "What is 3+3?",
        "What is 4+4?",
        "What is 5+5?"
    ]
    
    responses = await model.generate_batch(prompts)
    assert len(responses) == len(prompts)
    for response in responses:
        assert isinstance(response, str)
        assert len(response) > 0

@pytest.mark.asyncio
async def test_streaming():
    """Test streaming response generation."""
    model = LlamaModel(enable_streaming=True)
    
    # Test streaming response
    prompt = "Count from 1 to 5"
    chunks: List[str] = []
    
    async for chunk in model.generate_stream(prompt):
        chunks.append(chunk)
    
    assert len(chunks) > 0
    full_response = "".join(chunks)
    assert len(full_response) > 0

@pytest.mark.asyncio
async def test_token_counting():
    """Test token counting functionality."""
    model = LlamaModel()
    
    # Initial count should be 0
    assert model.get_token_count() == 0
    
    # Generate some responses
    await model.generate("Test prompt 1")
    count1 = model.get_token_count()
    assert count1 > 0
    
    await model.generate("Test prompt 2")
    count2 = model.get_token_count()
    assert count2 > count1
    
    # Test reset
    model.reset_token_count()
    assert model.get_token_count() == 0

@pytest.mark.asyncio
async def test_token_count_ttl():
    """Test token count TTL expiration."""
    model = LlamaModel(cache_ttl=1)  # 1 second TTL
    
    # Generate some responses
    await model.generate("Test prompt")
    initial_count = model.get_token_count()
    assert initial_count > 0
    
    # Wait for TTL to expire
    await asyncio.sleep(1.1)
    
    # Count should be reset
    assert model.get_token_count() == 0

@pytest.mark.asyncio
async def test_streaming_disabled():
    """Test behavior when streaming is disabled."""
    model = LlamaModel(enable_streaming=False)
    
    # Test streaming response (should return full response)
    prompt = "Test prompt"
    chunks: List[str] = []
    
    async for chunk in model.generate_stream(prompt):
        chunks.append(chunk)
    
    assert len(chunks) == 1
    assert len(chunks[0]) > 0

@pytest.mark.asyncio
async def test_llama_model_plan_generation():
    """Test that the model can generate a plan."""
    model = LlamaModel()
    task = "Navigate to example.com and click the login button"
    context = "The website is a simple demo site with a login form"
    
    plan = await model.generate_plan(task, context)
    assert isinstance(plan, list)
    assert len(plan) > 0
    assert all(isinstance(step, dict) for step in plan)

@pytest.mark.asyncio
async def test_llama_model_response_generation():
    """Test that the model can generate a response."""
    model = LlamaModel()
    task = "Navigate to example.com and click the login button"
    context = "The website is a simple demo site with a login form"
    plan = [
        {"type": "browser", "action": {"type": "navigate", "url": "https://example.com"}},
        {"type": "browser", "action": {"type": "click", "selector": "#login-button"}}
    ]
    
    response = await model.generate_response(task, plan, context)
    assert isinstance(response, str)
    assert len(response) > 0 