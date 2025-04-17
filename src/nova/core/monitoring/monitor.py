"""Agent Monitoring Implementation."""

from __future__ import annotations

import logging
import time
import psutil
import asyncio
from typing import Any, Dict, Optional
from datetime import datetime

from .config import MonitoringConfig

logger = logging.getLogger(__name__)

class AgentMonitor:
    """Monitors agent performance and behavior."""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        """Initialize the monitor."""
        self.config = config or MonitoringConfig()
        self._metrics: Dict[str, Any] = {}
        self._alerts: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        
    async def start(self) -> None:
        """Start the monitoring system."""
        if not self.config.enabled:
            return
            
        self._start_time = time.time()
        logger.info("Starting agent monitoring")
        
        # Start the metrics collection loop
        asyncio.create_task(self._collect_metrics())
        
    async def stop(self) -> None:
        """Stop the monitoring system."""
        if not self.config.enabled:
            return
            
        logger.info("Stopping agent monitoring")
        self._start_time = None
        
    async def _collect_metrics(self) -> None:
        """Collect system and agent metrics."""
        while self._start_time is not None:
            try:
                # Collect system metrics
                self._metrics["cpu_usage"] = psutil.cpu_percent()
                self._metrics["memory_usage"] = psutil.virtual_memory().percent
                
                # Check for alerts
                await self._check_alerts()
                
                # Sleep until next collection
                await asyncio.sleep(self.config.metrics_interval.total_seconds())
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
                
    async def _check_alerts(self) -> None:
        """Check metrics against alert thresholds."""
        for metric, threshold in self.config.alert_thresholds.items():
            if metric in self._metrics:
                value = self._metrics[metric]
                if value > threshold:
                    await self._trigger_alert(metric, value, threshold)
                    
    async def _trigger_alert(self, metric: str, value: float, threshold: float) -> None:
        """Trigger an alert for a metric exceeding its threshold."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "metric": metric,
            "value": value,
            "threshold": threshold,
        }
        
        self._alerts[metric] = alert
        logger.warning(f"Alert triggered: {metric} = {value} > {threshold}")
        
    def record_metric(self, name: str, value: Any) -> None:
        """Record a custom metric."""
        if not self.config.enabled:
            return
            
        self._metrics[name] = value
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return self._metrics.copy()
        
    def get_alerts(self) -> Dict[str, Any]:
        """Get all active alerts."""
        return self._alerts.copy()
        
    def clear_alerts(self) -> None:
        """Clear all active alerts."""
        self._alerts.clear()
        
    def get_uptime(self) -> float:
        """Get the monitor's uptime in seconds."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time 