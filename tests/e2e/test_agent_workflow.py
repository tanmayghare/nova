"""End-to-end tests for the Nova agent workflow."""

import asyncio
import logging
import os
import pytest
from datetime import datetime

from nova.core.agent import Agent
from nova.core.llm import LLM
from nova.core.memory import Memory
from nova.core.config import AgentConfig, BrowserConfig

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_agent_workflow():
    """Test the complete Nova agent workflow from dashboard to task completion."""
    
    # Initialize components
    llm = LLM()  # Using default configuration
    memory = Memory()
    
    # Configure agent
    agent_config = AgentConfig(
        max_parallel_tasks=2,
        browser_pool_size=1  # Reduced pool size for testing
    )
    
    browser_config = BrowserConfig(
        headless=True,  # Run in headless mode for testing
        viewport={"width": os.environ.get("BROWSER_VIEWPORT_WIDTH"), "height": os.environ.get("BROWSER_VIEWPORT_HEIGHT")}
    )
    
    # Create agent
    agent = Agent(
        llm=llm,
        memory=memory,
        config=agent_config,
        browser_config=browser_config
    )
    
    try:
        # Start the agent
        await agent.start()
        
        # Define the task
        task = {
            "name": "Test Task",
            "description": "Navigate to Nova Dashboard, create a new task, and execute it",
            "steps": [
                {
                    "type": "browser",
                    "action": {
                        "type": "navigate",
                        "url": "http://localhost:8000"  # Updated port
                    }
                },
                {
                    "type": "browser",
                    "action": {
                        "type": "click",
                        "selector": "button.create-task"
                    }
                },
                {
                    "type": "browser",
                    "action": {
                        "type": "type",
                        "selector": "input.task-name",
                        "text": "Test Automation Task"
                    }
                },
                {
                    "type": "browser",
                    "action": {
                        "type": "type",
                        "selector": "textarea.task-description",
                        "text": "This is a test task created by the agent"
                    }
                },
                {
                    "type": "browser",
                    "action": {
                        "type": "click",
                        "selector": "button.submit-task"
                    }
                }
            ]
        }
        
        # Execute the task
        task_id = str(datetime.now().timestamp())
        result = await agent.run(task, task_id)
        
        # Verify the results
        assert result is not None, "Task execution failed"
        assert isinstance(result, dict), f"Expected dictionary result, got {type(result)}"
        assert "status" in result, "Result missing status field"
        assert result["status"] == "completed", f"Task execution was not successful: {result}"
        assert "results" in result, "Result missing results field"
        assert len(result["results"]) > 0, "No results returned from task execution"
        
        # Check memory for task history
        context = await memory.get_context(task["description"])
        assert context is not None, "No context found in memory"
        assert len(context) > 0, "Empty context in memory"
        
        # Get performance metrics
        metrics = agent.get_performance_metrics()
        assert metrics is not None, "No performance metrics available"
        assert "task_count" in metrics, "Missing task count in metrics"
        assert metrics["task_count"] > 0, "No tasks recorded in metrics"
        
        logger.info(f"Task execution completed successfully. Result: {result}")
        logger.info(f"Performance metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise
    finally:
        # Clean up
        await agent.stop()
        
@pytest.mark.asyncio
async def test_parallel_task_execution():
    """Test parallel execution of multiple tasks."""
    
    # Initialize components
    llm = LLM()
    memory = Memory()
    
    # Configure agent for parallel execution
    agent_config = AgentConfig(
        max_parallel_tasks=3,
        browser_pool_size=1  # Reduced pool size for testing
    )
    
    browser_config = BrowserConfig(
        headless=True,  # Run in headless mode for testing
        viewport={"width": os.environ.get("BROWSER_VIEWPORT_WIDTH"), "height": os.environ.get("BROWSER_VIEWPORT_HEIGHT")}
    )
    
    # Create agent
    agent = Agent(
        llm=llm,
        memory=memory,
        config=agent_config,
        browser_config=browser_config
    )
    
    try:
        # Start the agent
        await agent.start()
        
        # Define multiple tasks
        tasks = [
            {
                "name": "Task 1",
                "description": "Navigate to dashboard and check status",
                "steps": [
                    {
                        "type": "browser",
                        "action": {
                            "type": "navigate",
                            "url": "http://localhost:8000"  # Updated port
                        }
                    },
                    {
                        "type": "browser",
                        "action": {
                            "type": "get_text",
                            "selector": ".status-indicator"
                        }
                    }
                ]
            },
            {
                "name": "Task 2",
                "description": "Create a new task with specific parameters",
                "steps": [
                    {
                        "type": "browser",
                        "action": {
                            "type": "navigate",
                            "url": "http://localhost:8000/tasks/new"  # Updated port
                        }
                    },
                    {
                        "type": "browser",
                        "action": {
                            "type": "type",
                            "selector": "input.task-name",
                            "text": "Parallel Test Task"
                        }
                    }
                ]
            },
            {
                "name": "Task 3",
                "description": "Check task history",
                "steps": [
                    {
                        "type": "browser",
                        "action": {
                            "type": "navigate",
                            "url": "http://localhost:8000/history"  # Updated port
                        }
                    },
                    {
                        "type": "browser",
                        "action": {
                            "type": "get_text",
                            "selector": ".task-list"
                        }
                    }
                ]
            }
        ]
        
        # Execute tasks in parallel
        tasks_to_run = []
        for task in tasks:
            task_id = str(datetime.now().timestamp())
            tasks_to_run.append(agent.run(task, task_id))
            
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks_to_run)
        
        # Verify results
        for i, result in enumerate(results):
            assert result is not None, f"Task {i} execution failed"
            assert isinstance(result, dict), f"Expected dictionary result for task {i}, got {type(result)}"
            assert "status" in result, f"Result for task {i} missing status field"
            assert result["status"] == "completed", f"Task {i} execution was not successful: {result}"
            assert "results" in result, f"Result for task {i} missing results field"
            assert len(result["results"]) > 0, f"No results returned from task {i} execution"
            
        # Check memory for all tasks
        for task in tasks:
            context = await memory.get_context(task["description"])
            assert context is not None, f"No context found for task: {task['description']}"
            
        # Get performance metrics
        metrics = agent.get_performance_metrics()
        assert metrics is not None, "No performance metrics available"
        assert metrics["task_count"] >= len(tasks), "Incorrect task count in metrics"
        
        logger.info(f"Parallel task execution completed successfully")
        logger.info(f"Performance metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise
    finally:
        # Clean up
        await agent.stop() 