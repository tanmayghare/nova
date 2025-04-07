"""Performance monitoring utilities."""

import time
import logging
from typing import Any, Callable, Dict, Optional
from functools import wraps

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor performance metrics for model operations."""
    
    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics: Dict[str, Dict[str, Any]] = {}
    
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