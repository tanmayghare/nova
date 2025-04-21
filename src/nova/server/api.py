"""API routes for the Nova Agent server."""

import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional
import base64
import os
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

# Import core agent components (adjust paths if necessary)
from ..core.llm import LLMConfig
from ..core.browser import BrowserConfig, Browser
from ..core.agents import TaskAgent, AgentState # Assuming TaskAgent takes config directly now
from ..core.agents import AgentConfig

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for active agents and their status/websockets
# WARNING: This is for MVP only. Not suitable for production (no persistence, scaling issues).
active_agents: Dict[str, TaskAgent] = {}
agent_websockets: Dict[str, List[WebSocket]] = {}
agent_status: Dict[str, Dict[str, Any]] = {} # Stores latest status/result

# --- Pydantic Models ---
class TaskRequest(BaseModel):
    task_description: str
    # Add other potential config overrides here later if needed

class TaskResponse(BaseModel):
    task_id: str
    message: str

class StatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    history: Optional[List[Dict]] = None

# --- Helper Functions ---
async def run_agent_task(task_id: str, agent: TaskAgent, task_description: str):
    """Runs the agent task and updates status/websockets."""
    logger.info(f"Running task '{task_description}' (ID: {task_id}) for agent...")
    agent_status[task_id] = {"status": "running", "result": None, "error": None, "history": []}
    await _broadcast_status(task_id, agent)
    
    try:
        # Run the agent with the actual task description and task_id
        final_outcome = await agent.run(task_id=task_id, task=task_description) 
        
        # Store final status
        agent_status[task_id] = {
            "status": final_outcome.get("status", "unknown"),
            "result": final_outcome.get("result"),
            "error": final_outcome.get("error"),
            "history": final_outcome.get("action_history", [])
        }
        logger.info(f"Task {task_id} finished with status: {agent_status[task_id]['status']}")
        # Broadcast final status *before* cleanup
        await _broadcast_status(task_id)

    except Exception as e:
        logger.error(f"Exception during agent run for task {task_id}: {e}", exc_info=True)
        agent_status[task_id]["status"] = "error"
        agent_status[task_id]["error"] = f"Agent run failed unexpectedly: {str(e)}"
        
    finally:
        # Clean up agent instance? Maybe browser needs explicit closing?
        # await agent.stop() # Add cleanup logic if needed
        if task_id in active_agents:
            del active_agents[task_id] # Remove agent from active list
        logger.info(f"Agent task {task_id} completed and cleaned up.")

async def _broadcast_status(task_id: str, agent: Optional[TaskAgent] = None):
    """Broadcasts the current status to all connected websockets for the task."""
    if task_id in agent_websockets:
        status_data = agent_status.get(task_id, {})
        screenshot_b64 = None
        current_url = None
        
        # Try to get current screenshot and URL from agent state if agent is provided
        # This requires agent state to be accessible, might need refactoring agent.run
        # For MVP, let's assume we don't have live state easily here yet, 
        # but add placeholder logic for when screenshot is part of the final_outcome perhaps.
        # A better approach would be for agent.run to yield/callback updates.
        # For now, we only send screenshot if it's part of the final stored status (e.g. on error)
        # TODO: Implement live state broadcasting from agent run loop

        if agent: # Check if agent instance is available (e.g., during initial broadcast)
             # Ideally, access live state: agent.state.latest_screenshot_bytes, agent.state.current_url
             # Placeholder: Check if screenshot was maybe stored in status on error? Unlikely.
             pass # Cannot access live state easily with current agent.run structure
        
        # If screenshot bytes are somehow available in the stored status_data (unlikely currently)
        screenshot_bytes = status_data.get("latest_screenshot_bytes")
        if screenshot_bytes:
            try:
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            except Exception as e:
                logger.error(f"Error base64 encoding screenshot for task {task_id}: {e}")
        
        # Similarly, get URL if stored
        current_url = status_data.get("current_url")

        status_payload = {
            "type": "status_update",
            "task_id": task_id,
            "status": status_data.get("status", "unknown"),
            "result": status_data.get("result"),
            "error": status_data.get("error"),
            "history_length": len(status_data.get("history", [])), # Send history length for brevity
            "screenshot_base64": screenshot_b64, # Add screenshot data
            "current_url": current_url # Add current URL
        }
        # Use gather to send to all websockets concurrently
        live_websockets = agent_websockets.get(task_id, [])
        if live_websockets:
            await asyncio.gather(*[ws.send_json(status_payload) for ws in live_websockets], return_exceptions=True)

# --- API Routes ---
@router.post("/start_task", response_model=TaskResponse)
async def start_task(request: TaskRequest):
    """Starts a new agent task."""
    task_id = str(uuid.uuid4())
    logger.info(f"Received request to start task {task_id}: {request.task_description}")
    
    try:
        # Initialize configurations (should read from .env)
        from dotenv import load_dotenv
        from pathlib import Path # Ensure Path is imported
        import os
        
        # Force override from .env file at project root
        dotenv_path = Path(__file__).resolve().parents[3] / '.env' # Corrected: parents[3]
        logger.info(f"DEBUG: Attempting to load .env from: {dotenv_path}")
        loaded = load_dotenv(dotenv_path=dotenv_path, override=True)
        logger.info(f"DEBUG: load_dotenv loaded={loaded}")
        
        # --- TEMPORARY DEBUG --- 
        service_type_env = os.environ.get("LLM_SERVICE_TYPE")
        logger.info(f"DEBUG: Read LLM_SERVICE_TYPE from env: '{service_type_env}'")
        # --- END DEBUG --- 

        # Create specific configs first
        llm_config = LLMConfig() 
        browser_config = BrowserConfig()
        
        # Create AgentConfig, embedding the LLMConfig
        # Provide dummy name/description for now, could be configurable later
        agent_config = AgentConfig(
             name="TaskExecutionAgent", 
             description="Agent for executing tasks via server.",
             llm_config=llm_config
             # memory_type, max_iterations, etc. will use defaults from AgentConfig definition
         )
        
        # Create the agent instance using AgentConfig
        agent = TaskAgent(config=agent_config, browser=Browser(browser_config))

        active_agents[task_id] = agent
        agent_status[task_id] = {"status": "pending", "result": None, "error": None, "history": []}
        
        # Run the agent task in the background, passing only task_id and agent
        # The task description is passed to agent.run() inside run_agent_task
        asyncio.create_task(run_agent_task(task_id, agent, request.task_description)) # Pass description here
        
        logger.info(f"Task {task_id} created and background execution started.")
        return TaskResponse(task_id=task_id, message="Task started successfully.")
        
    except Exception as e:
        logger.error(f"Failed to initialize or start task {task_id}: {e}", exc_info=True)
        # Clean up if partially created
        if task_id in active_agents:
            del active_agents[task_id]
        if task_id in agent_status:
            del agent_status[task_id]
        raise HTTPException(status_code=500, detail=f"Failed to start task: {str(e)}")

@router.get("/task_status/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str):
    """Gets the current status and result of a task."""
    if task_id not in agent_status:
        raise HTTPException(status_code=404, detail="Task ID not found")
        
    status_data = agent_status[task_id]
    return StatusResponse(
        task_id=task_id,
        status=status_data.get("status", "unknown"),
        result=status_data.get("result"),
        error=status_data.get("error"),
        history=status_data.get("history", [])
    )

@router.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time task updates."""
    await websocket.accept()
    logger.info(f"WebSocket connection accepted for task {task_id}")
    
    if task_id not in agent_websockets:
        agent_websockets[task_id] = []
    agent_websockets[task_id].append(websocket)
    
    # Send current status immediately upon connection
    current_status = agent_status.get(task_id, {"status": "not_found"})
    await websocket.send_json({
        "type": "status_update",
        "task_id": task_id,
        "status": current_status.get("status"),
        "result": current_status.get("result"),
        "error": current_status.get("error"),
        "history_length": len(current_status.get("history", []))
    })
    
    try:
        while True:
            # Keep the connection alive, listen for potential messages from client (optional)
            data = await websocket.receive_text() 
            logger.debug(f"Received message from WebSocket {task_id}: {data}")
            # Handle client messages if needed (e.g., pause/resume, provide input)
            await websocket.send_json({"type": "ack", "message": f"Received: {data}"})
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task {task_id}")
        agent_websockets[task_id].remove(websocket)
        if not agent_websockets[task_id]:
            del agent_websockets[task_id] # Clean up list if empty
    except Exception as e:
        logger.error(f"WebSocket error for task {task_id}: {e}", exc_info=True)
        # Ensure cleanup on error
        if task_id in agent_websockets and websocket in agent_websockets[task_id]:
             agent_websockets[task_id].remove(websocket)
             if not agent_websockets[task_id]:
                 del agent_websockets[task_id] 