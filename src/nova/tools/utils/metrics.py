"""Performance monitoring for tools."""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ToolMetric:
    """Metric data for a single tool execution."""
    tool_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    success: bool = False
    error: Optional[str] = None
    memory_usage: Optional[float] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None


class MetricsCollector:
    """Collects and manages performance metrics for tools."""
    
    def __init__(self) -> None:
        """Initialize the metrics collector."""
        self._metrics: Dict[str, List[ToolMetric]] = {}
        self._current_executions: Dict[str, ToolMetric] = {}
    
    def start_execution(self, tool_name: str) -> None:
        """Start tracking execution of a tool.
        
        Args:
            tool_name: Name of the tool being executed
        """
        metric = ToolMetric(
            tool_name=tool_name,
            start_time=datetime.now()
        )
        self._current_executions[tool_name] = metric
    
    def end_execution(
        self,
        tool_name: str,
        success: bool,
        error: Optional[str] = None,
        input_size: Optional[int] = None,
        output_size: Optional[int] = None
    ) -> None:
        """End tracking execution of a tool.
        
        Args:
            tool_name: Name of the tool
            success: Whether execution was successful
            error: Error message if execution failed
            input_size: Size of input data in bytes
            output_size: Size of output data in bytes
        """
        if tool_name not in self._current_executions:
            return
            
        metric = self._current_executions[tool_name]
        metric.end_time = datetime.now()
        metric.execution_time = (metric.end_time - metric.start_time).total_seconds()
        metric.success = success
        metric.error = error
        metric.input_size = input_size
        metric.output_size = output_size
        
        if tool_name not in self._metrics:
            self._metrics[tool_name] = []
        self._metrics[tool_name].append(metric)
        del self._current_executions[tool_name]
    
    def get_tool_metrics(self, tool_name: str) -> List[ToolMetric]:
        """Get metrics for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            List of metrics for the tool
        """
        return self._metrics.get(tool_name, [])
    
    def get_average_execution_time(self, tool_name: str) -> float:
        """Get average execution time for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Average execution time in seconds
        """
        metrics = self.get_tool_metrics(tool_name)
        if not metrics:
            return 0.0
        return sum(m.execution_time for m in metrics) / len(metrics)
    
    def get_success_rate(self, tool_name: str) -> float:
        """Get success rate for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Success rate as a percentage
        """
        metrics = self.get_tool_metrics(tool_name)
        if not metrics:
            return 0.0
        successful = sum(1 for m in metrics if m.success)
        return (successful / len(metrics)) * 100
    
    def clear_metrics(self, tool_name: Optional[str] = None) -> None:
        """Clear metrics for a tool or all tools.
        
        Args:
            tool_name: Name of the tool to clear metrics for, or None for all tools
        """
        if tool_name:
            self._metrics.pop(tool_name, None)
        else:
            self._metrics.clear() 