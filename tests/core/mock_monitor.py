"""Mock performance monitor implementation for testing."""

from typing import Dict, Any
from datetime import datetime, timedelta

class MockMonitor:
    """Mock performance monitor implementation for testing."""
    
    def __init__(self):
        """Initialize the mock monitor."""
        self._metrics = {
            "llm_calls": 0,
            "llm_latency": 0.0,
            "browser_actions": 0,
            "browser_latency": 0.0,
            "memory_operations": 0,
            "memory_latency": 0.0,
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0
        }
        self._start_time = datetime.now()
        
    def start_task(self, task_type: str) -> None:
        """Start tracking a task."""
        self._metrics["total_tasks"] += 1
        
    def end_task(self, task_type: str, success: bool = True) -> None:
        """End tracking a task."""
        if success:
            self._metrics["successful_tasks"] += 1
        else:
            self._metrics["failed_tasks"] += 1
            
    def record_llm_call(self, latency: float) -> None:
        """Record an LLM call."""
        self._metrics["llm_calls"] += 1
        self._metrics["llm_latency"] += latency
        
    def record_browser_action(self, latency: float) -> None:
        """Record a browser action."""
        self._metrics["browser_actions"] += 1
        self._metrics["browser_latency"] += latency
        
    def record_memory_operation(self, latency: float) -> None:
        """Record a memory operation."""
        self._metrics["memory_operations"] += 1
        self._metrics["memory_latency"] += latency
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        # Calculate averages
        if self._metrics["llm_calls"] > 0:
            self._metrics["avg_llm_latency"] = (
                self._metrics["llm_latency"] / self._metrics["llm_calls"]
            )
            
        if self._metrics["browser_actions"] > 0:
            self._metrics["avg_browser_latency"] = (
                self._metrics["browser_latency"] / self._metrics["browser_actions"]
            )
            
        if self._metrics["memory_operations"] > 0:
            self._metrics["avg_memory_latency"] = (
                self._metrics["memory_latency"] / self._metrics["memory_operations"]
            )
            
        # Calculate success rate
        if self._metrics["total_tasks"] > 0:
            self._metrics["success_rate"] = (
                self._metrics["successful_tasks"] / self._metrics["total_tasks"]
            )
            
        # Calculate uptime
        self._metrics["uptime"] = (
            datetime.now() - self._start_time
        ).total_seconds()
        
        return self._metrics
        
    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics = {
            "llm_calls": 0,
            "llm_latency": 0.0,
            "browser_actions": 0,
            "browser_latency": 0.0,
            "memory_operations": 0,
            "memory_latency": 0.0,
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0
        }
        self._start_time = datetime.now() 