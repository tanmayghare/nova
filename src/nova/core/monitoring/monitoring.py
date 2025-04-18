"""Performance monitoring utilities."""

import time
import logging
from typing import Any, Callable, Dict, Optional
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor performance metrics for model operations."""
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics = {
            "task_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "total_execution_time": 0,
            "average_execution_time": 0
        }
        self.start_time = None
        self.is_active = False
    
    def track(self, operation: str):
        """Decorator to track performance of a function.
        
        Args:
            operation: Name of the operation being tracked
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    # Update metrics
                    if operation not in self.metrics:
                        self.metrics[operation] = {
                            "count": 0,
                            "total_time": 0,
                            "avg_time": 0,
                            "min_time": float("inf"),
                            "max_time": 0
                        }
                    
                    metrics = self.metrics[operation]
                    metrics["count"] += 1
                    metrics["total_time"] += duration
                    metrics["avg_time"] = metrics["total_time"] / metrics["count"]
                    metrics["min_time"] = min(metrics["min_time"], duration)
                    metrics["max_time"] = max(metrics["max_time"], duration)
                    
                    logger.info(
                        f"Operation '{operation}' completed in {duration:.2f}s "
                        f"(avg: {metrics['avg_time']:.2f}s, "
                        f"min: {metrics['min_time']:.2f}s, "
                        f"max: {metrics['max_time']:.2f}s)"
                    )
                    
                    return result
                except Exception as e:
                    logger.error(f"Operation '{operation}' failed: {str(e)}")
                    raise
            return wrapper
        return decorator
    
    def get_metrics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics.
        
        Args:
            operation: Optional operation name to get metrics for
            
        Returns:
            Dictionary of performance metrics
        """
        if operation:
            return self.metrics.get(operation, {})
        return self.metrics
    
    def reset(self):
        """Reset all performance metrics."""
        self.metrics.clear()

    def start(self) -> None:
        """Start the performance monitor."""
        self.start_time = time.time()
        self.is_active = True
        logger.info("Performance monitoring started")
        
    def stop(self) -> None:
        """Stop the performance monitor."""
        if self.is_active:
            self.is_active = False
            if self.start_time:
                total_time = time.time() - self.start_time
                self.metrics["total_execution_time"] = total_time
            logger.info("Performance monitoring stopped")
        
    @contextmanager
    def track_task(self, task_id: str) -> None:
        """Track the execution of a task.
        
        Args:
            task_id: Identifier for the task
        """
        start_time = time.time()
        try:
            yield
            # Task completed successfully
            self.metrics["task_count"] += 1
            execution_time = time.time() - start_time
            self._update_metrics(execution_time, True)
        except Exception as e:
            # Task failed
            execution_time = time.time() - start_time
            self._update_metrics(execution_time, False)
            raise
            
    def _update_metrics(self, execution_time: float, success: bool) -> None:
        """Update performance metrics.
        
        Args:
            execution_time: Time taken to execute the task
            success: Whether the task was successful
        """
        # Update task count
        self.metrics["task_count"] += 1
        total_tasks = self.metrics["task_count"]
        
        # Update average execution time
        if total_tasks == 1:
            self.metrics["average_execution_time"] = execution_time
        else:
            current_avg = self.metrics["average_execution_time"]
            self.metrics["average_execution_time"] = (
                (current_avg * (total_tasks - 1) + execution_time) / total_tasks
            )
        
        # Update success rate
        if success:
            self.metrics["success_count"] += 1
        self.metrics["success_rate"] = (
            self.metrics["success_count"] / total_tasks
        ) * 100
        
        # Update failure rate
        current_failures = self.metrics["failure_count"]
        if not success:
            current_failures += 1
        self.metrics["failure_count"] = current_failures
        
        # Update total execution time
        self.metrics["total_execution_time"] += execution_time
        
        # Update success rate
        self.metrics["success_rate"] = self.metrics["success_count"] / total_tasks 