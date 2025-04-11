"""Web server implementation for Nova dashboard."""

import asyncio
import datetime
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from nova import __version__
from nova.web.api import router as api_router
from nova.web.monitoring import router as monitoring_router
from nova.web.monitoring import startup_event
from ..core.agent import Agent
from ..core.memory import Memory
from ..core.llm import LLM
from ..core.monitoring import PerformanceMonitor

logger = logging.getLogger(__name__)

# Get the path to the static and template directories
package_dir = Path(__file__).parent
static_dir = package_dir / "static"
templates_dir = package_dir / "templates"

# Global instances
agent: Optional[Agent] = None
memory: Optional[Memory] = None
llm: Optional[LLM] = None
monitor: Optional[PerformanceMonitor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI application."""
    # Startup
    logger.info("Starting Nova web server...")
    global agent, memory, llm, monitor
    
    # Initialize components
    memory = Memory()
    llm = LLM()
    monitor = PerformanceMonitor()
    agent = Agent(memory=memory, llm=llm)
    
    # Start agent and background tasks
    await agent.start()
    await startup_event()
    logger.info("Nova agent and background tasks started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Nova web server...")
    if agent:
        await agent.stop()
    logger.info("Nova agent stopped successfully")

# Create FastAPI app
app = FastAPI(
    title="Nova Dashboard",
    description="Web interface for Nova Agent",
    version=__version__,
    debug=True,  # Enable debug mode
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Set up templates
templates = Jinja2Templates(directory=str(templates_dir))

# Add custom Jinja2 filters
def now_filter(tz=None, fmt=None):
    return datetime.datetime.now(
        tz=datetime.timezone.utc if tz == "utc" else None
    ).strftime(fmt or "%Y-%m-%d %H:%M:%S")

templates.env.filters["now"] = now_filter

# Add API routers
app.include_router(api_router, prefix="/api")
app.include_router(monitoring_router, prefix="/monitoring")

# Add debug info to templates
@app.middleware("http")
async def add_debug_info(request: Request, call_next):
    """Add debug information to all templates."""
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        logger.error(f"Request error: {exc}", exc_info=True)
        return handle_exception(request, exc)

def handle_exception(request: Request, exc: Exception) -> JSONResponse:
    """Handle exceptions and return a detailed error response."""
    error_detail = {
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "request_path": str(request.url),
        "method": request.method,
    }
    
    logger.error(f"Error handling request: {error_detail}")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal Server Error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
    )

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the dashboard index page."""
    try:
        logger.info("Rendering index page")
        response = templates.TemplateResponse(
            "index.html", 
            {
                "request": request, 
                "version": __version__,
                "title": "Nova Dashboard",
            }
        )
        return response
    except Exception as e:
        logger.error(f"Error rendering index page: {e}", exc_info=True)
        raise


@app.get("/tasks", response_class=HTMLResponse)
async def tasks(request: Request):
    """Render the tasks page."""
    try:
        logger.info("Rendering tasks page")
        return templates.TemplateResponse(
            "tasks.html", 
            {
                "request": request, 
                "version": __version__,
                "title": "Tasks - Nova Dashboard",
            }
        )
    except Exception as e:
        logger.error(f"Error rendering tasks page: {e}", exc_info=True)
        raise


@app.get("/builder", response_class=HTMLResponse)
async def builder(request: Request):
    """Render the workflow builder page."""
    try:
        logger.info("Rendering builder page")
        return templates.TemplateResponse(
            "builder.html", 
            {
                "request": request, 
                "version": __version__,
                "title": "Workflow Builder - Nova Dashboard",
            }
        )
    except Exception as e:
        logger.error(f"Error rendering builder page: {e}", exc_info=True)
        raise


@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    """Render the settings page."""
    try:
        logger.info("Rendering settings page")
        return templates.TemplateResponse(
            "settings.html", 
            {
                "request": request, 
                "version": __version__,
                "title": "Settings - Nova Dashboard",
            }
        )
    except Exception as e:
        logger.error(f"Error rendering settings page: {e}", exc_info=True)
        raise


class TaskRequest(BaseModel):
    """Task request model."""
    description: str
    url: Optional[str] = None

@app.post("/api/tasks")
async def create_task(task: TaskRequest):
    """Create a new task."""
    if not agent:
        raise RuntimeError("Agent not initialized")
    
    task_id = await agent.create_task(
        description=task.description,
        url=task.url
    )
    return {"task_id": task_id}

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task status and results."""
    if not agent:
        raise RuntimeError("Agent not initialized")
    
    status = await agent.get_task_status(task_id)
    results = await agent.get_task_results(task_id)
    return {
        "status": status,
        "results": results
    }

@app.websocket("/ws/tasks/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time task updates."""
    if not agent:
        raise RuntimeError("Agent not initialized")
    
    await websocket.accept()
    try:
        while True:
            status = await agent.get_task_status(task_id)
            results = await agent.get_task_results(task_id)
            await websocket.send_json({
                "status": status,
                "results": results
            })
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

@app.get("/api/metrics")
async def get_metrics():
    """Get current performance metrics."""
    if not monitor:
        raise RuntimeError("Monitor not initialized")
    return monitor.get_metrics()

def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the Nova dashboard server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        reload: Whether to enable auto-reload
    """
    logger.info(f"Starting Nova dashboard on http://{host}:{port}")
    logger.info(f"Template directory: {templates_dir}")
    logger.info(f"Static directory: {static_dir}")
    
    # Check if template directory exists and is accessible
    if not templates_dir.exists():
        logger.error(f"Template directory does not exist: {templates_dir}")
        print(f"ERROR: Template directory does not exist: {templates_dir}")
        print("Creating basic template directory and files...")
        templates_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if static directory exists and is accessible
    if not static_dir.exists():
        logger.error(f"Static directory does not exist: {static_dir}")
        print(f"ERROR: Static directory does not exist: {static_dir}")
        print("Creating basic static directory...")
        static_dir.mkdir(parents=True, exist_ok=True)
    
    # List available templates
    if templates_dir.exists():
        templates_list = list(templates_dir.glob("*.html"))
        logger.info(f"Available templates: {[t.name for t in templates_list]}")
        print(f"Available templates: {[t.name for t in templates_list]}")
    
    try:
        uvicorn.run(
            "nova.web.server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="debug",  # Change to debug for more info
        )
    except Exception as e:
        logger.error(f"Error starting server: {e}", exc_info=True)
        print(f"ERROR: Failed to start server: {e}")
        raise 