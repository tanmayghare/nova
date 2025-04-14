"""Base classes for the enhanced tool system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolConfig(BaseModel):
    """Base configuration for tools."""
    name: str = Field(..., description="Name of the tool")
    version: str = Field("1.0.0", description="Version of the tool")
    description: str = Field(..., description="Description of the tool's functionality")
    enabled: bool = Field(True, description="Whether the tool is enabled")
    timeout: int = Field(30, description="Timeout in seconds for tool execution")
    retry_attempts: int = Field(3, description="Number of retry attempts on failure")
    dependencies: List[str] = Field(default_factory=list, description="List of tool dependencies")
    input_schema: Optional[Dict[str, Any]] = Field(default=None, description="Schema describing the expected input parameters")


class ToolResult:
    """Result of tool execution."""
    def __init__(
        self,
        success: bool,
        data: Any,
        error: Optional[str] = None,
        execution_time: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.data = data
        self.error = error
        self.execution_time = execution_time
        self.metadata = metadata or {}


class BaseTool(ABC):
    """Base class for all tools with enhanced capabilities."""
    
    # Class-level metrics collector
    _metrics = None

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize the tool with optional configuration."""
        # Import here to avoid circular imports
        from nova.tools.utils.metrics import MetricsCollector
        if BaseTool._metrics is None:
            BaseTool._metrics = MetricsCollector()
            
        self.config = config or self.get_default_config()
        self._validate_config()

    @classmethod
    @abstractmethod
    def get_default_config(cls) -> Any:
        """Get the default configuration for this tool."""
        pass

    def _validate_config(self) -> None:
        """Validate the tool configuration."""
        if not self.config.name:
            raise ValueError("Tool name is required")
        if not self.config.description:
            raise ValueError("Tool description is required")

    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate the input data before execution.
        
        Args:
            input_data: Dictionary containing input parameters
            
        Returns:
            bool indicating whether the input is valid
        """
        return True

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> ToolResult:
        """Execute the tool with the given input data.
        
        Args:
            input_data: Dictionary containing input parameters
            
        Returns:
            ToolResult containing execution results
        """
        pass

    async def run(self, input_data: Dict[str, Any]) -> ToolResult:
        """Run the tool with performance monitoring and advanced features.
        
        Args:
            input_data: Dictionary containing input parameters
            
        Returns:
            ToolResult containing execution results
        """
        # Start performance monitoring
        self._metrics.start_execution(self.config.name)
        
        try:
            # Validate input
            if not await self.validate_input(input_data):
                result = ToolResult(
                    success=False,
                    data=None,
                    error="Invalid input data",
                    execution_time=0.0
                )
                self._metrics.end_execution(
                    self.config.name,
                    success=False,
                    error="Invalid input data"
                )
                return result
            
            # Execute with timeout if configured
            if self.config.timeout:
                # TODO: Implement timeout logic
                pass
            
            # Execute tool
            result = await self.execute(input_data)
            
            # Update metrics
            self._metrics.end_execution(
                self.config.name,
                success=result.success,
                error=result.error,
                input_size=len(str(input_data)),
                output_size=len(str(result.data))
            )
            
            return result
            
        except Exception as e:
            error = str(e)
            self._metrics.end_execution(
                self.config.name,
                success=False,
                error=error
            )
            return ToolResult(
                success=False,
                data=None,
                error=error,
                execution_time=0.0
            )

    async def cleanup(self) -> None:
        """Clean up any resources used by the tool."""
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the tool.
        
        Returns:
            Dictionary containing tool metadata
        """
        return {
            "name": self.config.name,
            "version": self.config.version,
            "description": self.config.description,
            "enabled": self.config.enabled,
            "timeout": self.config.timeout,
            "retry_attempts": self.config.retry_attempts,
            "dependencies": list(self.config.dependencies),
            "metrics": {
                "avg_execution_time": self._metrics.get_average_execution_time(self.config.name),
                "success_rate": self._metrics.get_success_rate(self.config.name)
            }
        }

    @classmethod
    def get_metrics(cls) -> Any:
        """Get the metrics collector.
        
        Returns:
            The metrics collector instance
        """
        if cls._metrics is None:
            from nova.tools.utils.metrics import MetricsCollector
            cls._metrics = MetricsCollector()
        return cls._metrics 