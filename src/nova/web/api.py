"""API endpoints for the Nova dashboard."""

import logging
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

from nova import __version__
from ..core.agent import Agent
from ..core.browser import Browser
from ..core.llm import LLM
from ..core.config import BrowserConfig

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(tags=["api"])

# Task storage
TASKS_FILE = os.path.join("outputs", "tasks.json")
tasks: Dict[str, Dict[str, Any]] = {}
active_agents: Dict[str, Agent] = {}

def load_tasks():
    """Load tasks from file."""
    global tasks
    if os.path.exists(TASKS_FILE):
        try:
            with open(TASKS_FILE, "r") as f:
                data = json.load(f)
                # Convert list to dictionary if needed
                if isinstance(data, list):
                    tasks = {}
                    for i, task in enumerate(data):
                        tasks[f"task_{i}"] = task
                else:
                    tasks = data
        except Exception as e:
            logger.error(f"Error loading tasks: {e}")
            tasks = {}
    else:
        logger.info(f"Tasks file not found, creating new one at {TASKS_FILE}")
        tasks = {}
        save_tasks()

def save_tasks():
    """Save tasks to file."""
    try:
        with open(TASKS_FILE, "w") as f:
            json.dump(tasks, f, default=str)
    except Exception as e:
        logger.error(f"Error saving tasks: {e}")

# Load tasks on startup
load_tasks()


class TaskRequest(BaseModel):
    """Model for task execution requests."""

    task: str = Field(..., description="Task description")
    model: str = Field("mistral-small3.1:24b-instruct-2503-q4_K_M", description="LLM model to use")
    headless: bool = Field(False, description="Run browser in headless mode")


class TaskResponse(BaseModel):
    """Model for task execution responses."""

    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status")
    created_at: datetime = Field(..., description="Task creation timestamp")


class TaskStatus(BaseModel):
    """Model for task status responses."""

    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status")
    created_at: datetime = Field(..., description="Task creation timestamp")
    result: Optional[Any] = Field(None, description="Task result")
    error: Optional[str] = Field(None, description="Error message if task failed")

    @classmethod
    def from_task_data(cls, task_data: Dict[str, Any]) -> "TaskStatus":
        """Create a TaskStatus instance from task data.
        
        Args:
            task_data: Raw task data dictionary
            
        Returns:
            TaskStatus instance with parsed result
        """
        result = task_data.get("result")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                pass  # Keep as string if not valid JSON
                
        return cls(
            task_id=task_data["task_id"],
            status=task_data["status"],
            created_at=task_data["created_at"],
            result=result,
            error=task_data.get("error")
        )


@router.get("/info")
async def get_info() -> Dict[str, Any]:
    """Get Nova information."""
    return {
        "name": "Nova",
        "version": __version__,
        "description": "An intelligent browser automation agent",
    }


@router.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(task_request: TaskRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Create a new task.
    
    Args:
        task_request: Task execution request
        background_tasks: Background tasks handler
        
    Returns:
        Task information
    """
    # Generate task ID
    task_id = f"task_{len(tasks) + 1}_{int(datetime.now().timestamp())}"
    
    # Create task record
    tasks[task_id] = {
        "task_id": task_id,
        "task": task_request.task,
        "model": task_request.model,
        "headless": task_request.headless,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "result": None,
        "error": None,
        "steps": [],  # Track individual steps
        "llm_response": None,  # Store LLM response
    }
    
    # Save tasks
    save_tasks()
    
    # Schedule task for execution
    background_tasks.add_task(execute_task, task_id, task_request)
    
    return {
        "task_id": task_id,
        "status": "pending",
        "created_at": tasks[task_id]["created_at"],
    }


@router.get("/tasks", response_model=List[TaskStatus])
async def list_tasks() -> List[Dict[str, Any]]:
    """List all tasks.
    
    Returns:
        List of tasks
    """
    # Ensure tasks is a dictionary
    if isinstance(tasks, list):
        return []
    return list(tasks.values())


@router.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task(task_id: str) -> Dict[str, Any]:
    """Get task status.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Task status
        
    Raises:
        HTTPException: If task not found
    """
    if task_id not in tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )
    
    return TaskStatus.from_task_data(tasks[task_id]).dict()


@router.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(task_id: str) -> None:
    """Delete a task.
    
    Args:
        task_id: Task identifier
        
    Raises:
        HTTPException: If task not found
    """
    if task_id not in tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )
    
    # Stop agent if running
    if task_id in active_agents:
        await active_agents[task_id].stop()
        del active_agents[task_id]
    
    # Remove task
    del tasks[task_id]


async def execute_task(task_id: str, task_request: TaskRequest) -> None:
    """Execute a task in the background.
    
    Args:
        task_id: Task identifier
        task_request: Task request details
    """
    browser = None
    try:
        # Initialize browser
        browser = Browser(BrowserConfig(headless=task_request.headless))
        await browser.start()
        
        # Initialize agent
        agent = Agent(
            llm=LLM(model_name=task_request.model),
            browser=browser,
        )
        
        # Store agent reference
        active_agents[task_id] = agent
        
        # Update task status
        tasks[task_id]["status"] = "running"
        save_tasks()
        
        # Execute task
        result = await agent.run(task_request.task)
        
        # Get LLM response
        try:
            llm_response = agent.llm.get_last_response()
        except AttributeError:
            logger.warning("LLM response not available, using result as response")
            llm_response = str(result)
        
        # Update task status and result
        tasks[task_id].update({
            "status": "completed",
            "result": str(result),
            "llm_response": llm_response,
            "completed_at": datetime.now().isoformat(),
        })
        save_tasks()
        
        # Record in execution history
        from nova.web.monitoring import add_execution_record
        add_execution_record(
            task=task_request.task,
            result=str(result),
        )
        
    except Exception as e:
        # Update task status with error
        tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat(),
        })
        save_tasks()
        
        # Record error in execution history
        from nova.web.monitoring import add_execution_record
        add_execution_record(
            task=task_request.task,
            error=str(e),
        )
        
        logger.error(f"Task execution failed: {e}", exc_info=True)
    finally:
        # Clean up
        if task_id in active_agents:
            agent = active_agents.pop(task_id)
            try:
                if agent.browser:
                    await agent.browser.stop()
            except Exception as e:
                logger.error(f"Error stopping browser: {e}", exc_info=True)
        
        # Force stop browser if it's still open
        if browser:
            try:
                await browser.stop()
            except Exception as e:
                logger.error(f"Error force stopping browser: {e}", exc_info=True) 