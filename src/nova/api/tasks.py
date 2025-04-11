import json
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..core.task import Task
import time

logger = logging.getLogger(__name__)

class TaskRequest(BaseModel):
    task: str
    model: Optional[str] = "llama3.2:3b-instruct-q8_0"
    headless: Optional[bool] = False

class TaskResponse(BaseModel):
    task_id: str
    status: str
    created_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

router = APIRouter()

@router.post("/tasks")
async def create_task(request: TaskRequest) -> TaskResponse:
    """Create a new task."""
    try:
        # Generate task ID
        task_id = f"task_1_{int(time.time())}"
        
        # Create and execute task
        task = Task(
            task_id=task_id,
            description=request.task,
            model=request.model,
            headless=request.headless
        )
        
        # Execute task asynchronously
        result = await task.execute()
        return TaskResponse(**result)
        
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> TaskResponse:
    """Get task status and result."""
    try:
        task = Task(
            task_id=task_id,
            description="",  # Not needed for status check
            model="",  # Not needed for status check
            headless=False  # Not needed for status check
        )
        result = task._to_dict()
        
        # Ensure result is properly formatted
        if isinstance(result.get("result"), str):
            try:
                result["result"] = json.loads(result["result"])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse result as JSON: {str(e)}")
                result["result"] = {"error": f"Invalid result format: {str(e)}"}

        # Ensure error is properly formatted
        if isinstance(result.get("error"), str):
            try:
                result["error"] = json.loads(result["error"])
            except json.JSONDecodeError:
                # Keep the error as a string if it's not valid JSON
                pass

        return TaskResponse(**result)
    except Exception as e:
        logger.error(f"Error getting task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 