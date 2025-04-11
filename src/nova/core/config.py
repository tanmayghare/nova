from typing import Dict, List, Optional

from pydantic import BaseModel


class BrowserConfig(BaseModel):
    """Configuration for the browser."""

    headless: bool = True
    viewport: Optional[Dict[str, int]] = None
    browser_args: List[str] = []


class AgentConfig(BaseModel):
    """Configuration for the agent."""

    max_steps: int = 100
    max_failures: int = 3
    retry_delay: int = 5
    use_vision: bool = True
    enable_memory: bool = True 