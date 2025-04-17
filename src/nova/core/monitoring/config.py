"""Agent Monitoring Configuration."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import timedelta

@dataclass
class MonitoringConfig:
    """Configuration for agent monitoring.
    
    Attributes:
        enabled: Whether monitoring is enabled
        metrics_interval: How often to collect metrics
        log_level: Logging level for monitoring
        storage_backend: Where to store monitoring data
        storage_config: Configuration for the storage backend
        alert_thresholds: Thresholds for triggering alerts
    """
    
    enabled: bool = True
    metrics_interval: timedelta = timedelta(seconds=60)
    log_level: str = "INFO"
    storage_backend: str = "memory"  # Options: memory, file, database
    storage_config: Optional[Dict[str, Any]] = None
    alert_thresholds: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.storage_config is None:
            self.storage_config = {}
            
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "cpu_usage": 80.0,  # Percentage
                "memory_usage": 80.0,  # Percentage
                "response_time": 5.0,  # Seconds
                "error_rate": 0.1,  # Percentage
            } 