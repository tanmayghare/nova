"""End-to-end tests for the agent workflow."""

import os
import json
import asyncio
import pytest
from typing import Dict, List, Any
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from nova.core.agent import Agent
from nova.core.config import AgentConfig, BrowserConfig
from tests.core.mock_memory import MockMemory
from tests.core.mock_llm import MockLLM
from tests.core.mock_browser import MockBrowser
from tests.core.mock_monitor import MockMonitor

# Configure logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
async def agent():
    """Create an agent instance for testing."""
    memory = MockMemory()
    agent = Agent(memory=memory)
    yield agent
    await memory.clear()

@pytest.mark.asyncio
async def test_complete_workflow(agent):
    """Test complete agent workflow."""
    # Setup test data
    task = "Test task"
    context = {"key": "value"}
    
    # Execute task
    result = await agent.execute_task(task, context)
    
    # Verify memory storage
    memories = await agent.memory.get_memories(limit=1)
    assert len(memories) == 1
    assert memories[0]["content"] == result
    
    # Verify performance metrics
    metrics = agent.get_performance_metrics()
    assert "execution_time" in metrics
    assert "memory_usage" in metrics

@pytest.mark.asyncio
async def test_memory_context_retrieval(agent):
    """Test memory context retrieval."""
    # Add test memories
    test_content = "Test memory content"
    await agent.memory.add_memory(test_content)
    
    # Retrieve context
    context = await agent.memory.get_context("test memory")
    assert len(context) > 0
    assert context[0]["content"] == test_content

@pytest.mark.asyncio
async def test_llm_caching(agent):
    """Test LLM response caching."""
    # Execute same task twice
    task = "Cache test task"
    result1 = await agent.execute_task(task)
    result2 = await agent.execute_task(task)
    
    # Verify cache hit
    assert result1 == result2
    metrics = agent.get_performance_metrics()
    assert metrics["cache_hits"] > 0

@pytest.mark.asyncio
async def test_error_handling(agent):
    """Test error handling."""
    # Trigger an error condition
    with pytest.raises(Exception):
        await agent.execute_task(None)
    
    # Verify error is logged in memory
    memories = await agent.memory.get_memories()
    assert any("error" in str(m["content"]).lower() for m in memories)

@pytest.mark.asyncio
async def test_performance_monitoring(agent):
    """Test performance monitoring."""
    # Execute a task
    await agent.execute_task("Performance test")
    
    # Verify metrics
    metrics = agent.get_performance_metrics()
    assert metrics["execution_time"] > 0
    assert metrics["memory_usage"] > 0
    assert "api_calls" in metrics

@pytest.mark.asyncio
async def test_concurrent_operations(agent):
    """Test concurrent task execution."""
    # Execute multiple tasks concurrently
    tasks = ["Task 1", "Task 2", "Task 3"]
    results = await asyncio.gather(
        *[agent.execute_task(task) for task in tasks]
    )
    
    # Verify all tasks completed
    assert len(results) == len(tasks)
    
    # Verify memory entries
    memories = await agent.memory.get_memories()
    assert len(memories) >= len(tasks)

if __name__ == "__main__":
    pytest.main([__file__]) 