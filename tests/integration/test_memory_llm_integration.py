import pytest
from unittest.mock import AsyncMock, MagicMock
import json

from nova.core.memory import Memory
from nova.core.llm import LangChainAdapter
from langchain_core.language_models.chat_models import BaseChatModel


@pytest.fixture
def mock_model():
    """Create a mock LangChain model."""
    model = MagicMock(spec=BaseChatModel)
    model.ainvoke = AsyncMock()
    return model


@pytest.fixture
async def memory():
    """Create a memory instance with some pre-populated memories."""
    memory = Memory()
    
    # Add some memories
    await memory.add(
        "navigate to example.com",
        {"type": "browser", "action": {"type": "navigate", "url": "https://example.com"}},
        "Successfully navigated to example.com"
    )
    
    await memory.add(
        "search for python",
        {"type": "browser", "action": {"type": "type", "selector": "#search", "text": "python"}},
        "Entered 'python' in search box"
    )
    
    return memory


@pytest.mark.asyncio
async def test_plan_generation_with_memory_context(mock_model, memory):
    """Test that the LLM receives context from memory when generating plans."""
    # Setup mock response
    mock_response = MagicMock()
    mock_response.content = """[
        {"type": "browser", "action": {"type": "navigate", "url": "https://example.com/search"}}
    ]"""
    mock_model.ainvoke.return_value = mock_response
    
    # Create LangChain adapter
    adapter = LangChainAdapter(mock_model)
    
    # Get context from memory
    task = "search for python on example.com"
    context = await memory.get_context(task)
    
    # Generate plan with context
    plan = await adapter.generate_plan(task, context)
    
    # Verify the plan was generated correctly
    assert len(plan) == 1
    assert plan[0]["type"] == "browser"
    
    # Verify that the memory context was included in the prompt
    # Extract the prompt from the ainvoke call
    call_args = mock_model.ainvoke.call_args[0][0]
    assert "Context:" in call_args
    assert "navigate to example.com" in call_args or "search for python" in call_args


@pytest.mark.asyncio
async def test_memory_update_after_plan_execution(mock_model, memory):
    """Test memory updates after plan execution."""
    # Setup mock response for generate_response
    mock_response = MagicMock()
    mock_response.content = "Successfully executed the search."
    mock_model.ainvoke.return_value = mock_response
    
    # Create LangChain adapter
    adapter = LangChainAdapter(mock_model)
    
    # Define task and executed plan
    task = "search for python libraries"
    executed_plan = [
        {
            "type": "browser", 
            "action": {"type": "navigate", "url": "https://pypi.org"},
            "result": "Navigated to PyPI"
        },
        {
            "type": "browser",
            "action": {"type": "type", "selector": "#search", "text": "python libraries"},
            "result": "Entered search term"
        }
    ]
    
    # Get context from memory
    context = await memory.get_context(task)
    
    # Generate response
    response = await adapter.generate_response(task, executed_plan, context)
    
    # Add to memory
    for step in executed_plan:
        await memory.add(task, step, step["result"])
    await memory.add(task, {"type": "summary"}, response)
    
    # Verify memory has been updated
    new_context = await memory.get_context(task)
    # The implementation includes the results, not the task name
    assert "Navigated to PyPI" in new_context
    assert "Entered search term" in new_context
    assert "Successfully executed the search" in new_context
    
    # Verify summary contains info
    summary = await memory.get_summary(task)
    assert "Navigated to PyPI" in summary
    assert "Entered search term" in summary
    assert "Successfully executed the search" in summary


@pytest.mark.asyncio
async def test_plan_adaptation_based_on_memory(mock_model, memory):
    """Test that plans adapt based on memory feedback."""
    # First, simulate a failed attempt
    failed_task = "login to website"
    await memory.add(
        failed_task,
        {"type": "browser", "action": {"type": "click", "selector": "#login"}},
        "Error: selector #login not found"
    )
    
    # Setup different responses based on prompt content
    def simulate_learning(prompt):
        if "Error: selector #login not found" in prompt:
            # The model should "learn" from the previous error
            return MagicMock(content="""[
                {"type": "browser", "action": {"type": "click", "selector": ".login-button"}}
            ]""")
        return MagicMock(content="""[
            {"type": "browser", "action": {"type": "click", "selector": "#login"}}
        ]""")
    
    mock_model.ainvoke.side_effect = simulate_learning
    
    # Create LangChain adapter
    adapter = LangChainAdapter(mock_model)
    
    # Get context that includes the error
    context = await memory.get_context(failed_task)
    assert "Error: selector #login not found" in context
    
    # Generate new plan with context from failed attempt
    new_plan = await adapter.generate_plan(failed_task, context)
    
    # Verify the new plan uses a different selector
    assert new_plan[0]["action"]["selector"] == ".login-button"


@pytest.mark.asyncio
async def test_memory_serialization_with_plan_data(mock_model):
    """Test serialization of memory with complex plan data."""
    memory = Memory()
    
    # Add a complex memory entry
    complex_step = {
        "type": "browser",
        "action": {
            "type": "fill_form",
            "fields": [
                {"selector": "#username", "value": "testuser"},
                {"selector": "#password", "value": "******"}
            ]
        }
    }
    
    await memory.add("login", complex_step, "Form filled successfully")
    
    # Serialize and deserialize
    json_str = memory.to_json()
    new_memory = Memory.from_json(json_str)
    
    # Check if complex data structure is preserved
    assert new_memory._memory[0]["step"]["action"]["fields"][0]["selector"] == "#username"
    assert new_memory._memory[0]["step"]["action"]["fields"][1]["value"] == "******"
    
    # Test using the restored memory with LLM
    mock_response = MagicMock()
    mock_response.content = "Response generated"
    mock_model.ainvoke.return_value = mock_response
    
    adapter = LangChainAdapter(mock_model)
    context = await new_memory.get_context("login")
    response = await adapter.generate_response("login", [new_memory._memory[0]["step"]], context)
    
    assert response == "Response generated" 