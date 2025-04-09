"""Advanced configuration management for tools."""

import os
import json
import yaml
from typing import Any, Dict, Optional, Set
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AdvancedToolConfig:
    """Enhanced configuration for tools with advanced features."""
    name: str
    version: str
    description: str
    enabled: bool = True
    timeout: int = 30
    retry_attempts: int = 3
    dependencies: Set[str] = field(default_factory=set)
    
    # Advanced configuration options
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None
    log_level: str = "INFO"
    cache_enabled: bool = False
    cache_ttl: int = 3600  # seconds
    rate_limit: Optional[int] = None  # requests per minute
    concurrent_executions: int = 1
    environment_vars: Dict[str, str] = field(default_factory=dict)
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class ConfigurationManager:
    """Advanced configuration manager with multiple storage backends."""
    
    def __init__(
        self,
        config_dir: Optional[str] = None,
        format: str = "yaml"
    ) -> None:
        """Initialize the configuration manager.
        
        Args:
            config_dir: Directory to store configuration files
            format: Configuration file format ('yaml' or 'json')
        """
        self.config_dir = Path(config_dir or os.path.expanduser("~/.nova/config"))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.format = format
        self._configs: Dict[str, AdvancedToolConfig] = {}
        self._load_all_configs()
    
    def _load_all_configs(self) -> None:
        """Load all configuration files from the config directory."""
        pattern = f"*.{self.format}"
        for config_file in self.config_dir.glob(pattern):
            try:
                config = self._load_config_file(config_file)
                if config:
                    self._configs[config.name] = config
            except Exception as e:
                print(f"Error loading config {config_file}: {e}")
    
    def _load_config_file(self, path: Path) -> Optional[AdvancedToolConfig]:
        """Load a configuration file.
        
        Args:
            path: Path to the configuration file
            
        Returns:
            Loaded configuration or None if loading failed
        """
        try:
            with open(path, 'r') as f:
                if self.format == 'yaml':
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            return AdvancedToolConfig(**data)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def _save_config_file(self, config: AdvancedToolConfig) -> None:
        """Save a configuration to file.
        
        Args:
            config: Configuration to save
        """
        path = self.config_dir / f"{config.name}.{self.format}"
        data = {
            field: getattr(config, field)
            for field in config.__dataclass_fields__
        }
        
        try:
            with open(path, 'w') as f:
                if self.format == 'yaml':
                    yaml.safe_dump(data, f)
                else:
                    json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving {path}: {e}")
    
    def get_config(self, tool_name: str) -> Optional[AdvancedToolConfig]:
        """Get configuration for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool configuration if found
        """
        return self._configs.get(tool_name)
    
    def set_config(self, config: AdvancedToolConfig) -> None:
        """Set configuration for a tool.
        
        Args:
            config: Tool configuration
        """
        self._configs[config.name] = config
        self._save_config_file(config)
    
    def update_config(self, tool_name: str, updates: Dict[str, Any]) -> None:
        """Update configuration for a tool.
        
        Args:
            tool_name: Name of the tool
            updates: Dictionary of configuration updates
        """
        config = self.get_config(tool_name)
        if not config:
            return
            
        for key, value in updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.set_config(config)
    
    def delete_config(self, tool_name: str) -> None:
        """Delete configuration for a tool.
        
        Args:
            tool_name: Name of the tool
        """
        if tool_name in self._configs:
            del self._configs[tool_name]
            path = self.config_dir / f"{tool_name}.{self.format}"
            if path.exists():
                path.unlink()
    
    def list_configs(self) -> Dict[str, AdvancedToolConfig]:
        """List all tool configurations.
        
        Returns:
            Dictionary of tool configurations
        """
        return self._configs.copy() 