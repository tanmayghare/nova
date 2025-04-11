"""Monitoring endpoints for the Nova dashboard."""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psutil
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from nova import __version__

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(tags=["monitoring"])

# WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Agent execution history
execution_history: List[Dict[str, Any]] = []

# Connection manager for WebSockets
class ConnectionManager:
    """WebSocket connection manager."""
    
    def __init__(self) -> None:
        """Initialize the connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_count = 0
        self.stats_task: Optional[asyncio.Task] = None
    
    async def connect(self, websocket: WebSocket) -> str:
        """Connect a client to the websocket.
        
        Args:
            websocket: The WebSocket instance
            
        Returns:
            Connection ID
        """
        await websocket.accept()
        # Generate a unique ID for this connection
        conn_id = str(uuid.uuid4())
        self.active_connections[conn_id] = websocket
        self.connection_count += 1
        logger.info(f"WebSocket client connected: {conn_id} (Total: {self.connection_count})")
        
        # Send initial data to client
        try:
            stats = await get_stats()
            await websocket.send_text(json.dumps({
                "type": "stats",
                "data": stats,
                "timestamp": datetime.now().isoformat(),
                "message": "Connected successfully",
            }))
        except Exception as e:
            logger.error(f"Error sending initial data to {conn_id}: {e}")
        
        # Start stats broadcasting if not already running
        if not self.stats_task or self.stats_task.done():
            self.stats_task = asyncio.create_task(self.broadcast_stats())
        
        return conn_id
    
    def disconnect(self, conn_id: str) -> None:
        """Disconnect a client.
        
        Args:
            conn_id: Connection ID
        """
        if conn_id in self.active_connections:
            del self.active_connections[conn_id]
            self.connection_count -= 1
            logger.info(f"WebSocket client disconnected: {conn_id} (Total: {self.connection_count})")
    
    async def send_message(self, conn_id: str, message: str) -> bool:
        """Send a message to a specific client.
        
        Args:
            conn_id: Connection ID
            message: JSON string message
            
        Returns:
            True if successful, False otherwise
        """
        if conn_id in self.active_connections:
            try:
                await self.active_connections[conn_id].send_text(message)
                return True
            except Exception as e:
                logger.error(f"Error sending message to {conn_id}: {e}")
                # Clean up the connection if it's dead
                self.disconnect(conn_id)
                return False
        return False
    
    async def broadcast(self, message: str) -> Tuple[int, int]:
        """Broadcast a message to all connected clients.
        
        Args:
            message: JSON string message
            
        Returns:
            Tuple of (success_count, failure_count)
        """
        success_count = 0
        failure_count = 0
        
        # Make a copy since we might modify the dictionary during iteration
        connections = dict(self.active_connections)
        
        for conn_id, connection in connections.items():
            try:
                await connection.send_text(message)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to send to {conn_id}: {e}")
                self.disconnect(conn_id)
                failure_count += 1
        
        return success_count, failure_count
    
    async def broadcast_stats(self) -> None:
        """Broadcast system stats to all connected clients."""
        logger.info("Starting stats broadcast task")
        try:
            while self.active_connections:
                # Get current stats
                stats = await get_stats()
                
                message = json.dumps({
                    "type": "stats",
                    "data": stats,
                    "timestamp": datetime.now().isoformat(),
                })
                
                # Broadcast to all connections
                success, failed = await self.broadcast(message)
                
                if failed > 0:
                    logger.warning(f"Failed to send stats to {failed} clients")
                
                # Stop broadcasting if no clients are connected
                if not self.active_connections:
                    logger.info("No active connections, stopping stats broadcast")
                    break
                
                # Wait before sending next update
                await asyncio.sleep(2)
        except asyncio.CancelledError:
            logger.info("Stats broadcast task cancelled")
        except Exception as e:
            logger.error(f"Error in stats broadcast task: {e}", exc_info=True)
        
        logger.info("Stats broadcast task stopped")

# Create a global connection manager
manager = ConnectionManager()


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Get system and Nova stats.
    
    Returns:
        System and Nova statistics
    """
    try:
        process = psutil.Process(os.getpid())
        
        return {
            "system": {
                "cpu_usage": psutil.cpu_percent(interval=0.1),
                "memory_usage": psutil.virtual_memory().percent,
                "memory_available": psutil.virtual_memory().available / (1024 * 1024),  # MB
            },
            "process": {
                "cpu_usage": process.cpu_percent(interval=0.1),
                "memory_usage": process.memory_info().rss / (1024 * 1024),  # MB
                "threads": process.num_threads(),
            },
            "nova": {
                "version": __version__,
                "uptime": int(time.time() - process.create_time()),
                "execution_count": len(execution_history),
                "active_websockets": manager.connection_count,
            },
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        # Return basic stats in case of error
        return {
            "system": {"cpu_usage": 0, "memory_usage": 0, "memory_available": 0},
            "process": {"cpu_usage": 0, "memory_usage": 0, "threads": 0},
            "nova": {"version": __version__, "uptime": 0, "execution_count": 0, "error": str(e)},
        }


@router.get("/history")
async def get_history() -> List[Dict[str, Any]]:
    """Get agent execution history.
    
    Returns:
        List of execution history entries
    """
    return execution_history


@router.get("/connections")
async def get_connections() -> Dict[str, Any]:
    """Get active WebSocket connections.
    
    Returns:
        Connection information
    """
    return {
        "active_connections": len(manager.active_connections),
        "connection_ids": list(manager.active_connections.keys()),
    }


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle WebSocket connections for real-time monitoring.
    
    Args:
        websocket: WebSocket connection
    """
    conn_id = await manager.connect(websocket)
    
    try:
        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()
            
            # Log heartbeats at debug level to avoid spamming logs
            if data == "heartbeat":
                logger.debug(f"Heartbeat from {conn_id}")
            else:
                logger.info(f"Received message from {conn_id}: {data}")
                
                # Echo back any non-heartbeat messages
                await manager.send_message(conn_id, json.dumps({
                    "type": "echo",
                    "data": data,
                    "timestamp": datetime.now().isoformat(),
                }))
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected normally: {conn_id}")
        manager.disconnect(conn_id)
    except Exception as e:
        logger.error(f"WebSocket error for {conn_id}: {e}", exc_info=True)
        manager.disconnect(conn_id)
        
        # Try to send close message if possible
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason=str(e))
        except Exception:
            pass


def add_execution_record(task: str, result: Optional[str] = None, error: Optional[str] = None) -> None:
    """Add a record to execution history.
    
    Args:
        task: Task description
        result: Task result (if successful)
        error: Error message (if failed)
    """
    try:
        execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "status": "success" if result else "error",
            "result": result,
            "error": error,
        })
        
        # Keep history size reasonable
        if len(execution_history) > 100:
            execution_history.pop(0)
            
        # Notify clients about the new record asynchronously
        asyncio.create_task(notify_clients_about_execution())
    except Exception as e:
        logger.error(f"Error adding execution record: {e}", exc_info=True)


async def notify_clients_about_execution() -> None:
    """Notify WebSocket clients about new execution record."""
    if not execution_history:
        return
    
    latest = execution_history[-1]
    message = json.dumps({
        "type": "execution",
        "data": latest,
        "timestamp": datetime.now().isoformat(),
    })
    
    await manager.broadcast(message)


# Startup function to initialize background tasks
async def startup_event() -> None:
    """Start background tasks on startup."""
    logger.info("Initializing monitoring background tasks")
    # The broadcast task will be created when the first client connects 