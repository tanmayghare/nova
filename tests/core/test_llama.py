"""Tests for Llama model integration."""

import pytest
from nova.core.llama import LlamaModel

@pytest.mark.asyncio
async def test_llama_model_initialization():
    """Test that the Llama model initializes correctly."""
    model = LlamaModel()
    assert model is not None

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