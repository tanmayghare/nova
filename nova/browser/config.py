from typing import Dict, List, Optional

from pydantic import BaseModel


class BrowserConfig(BaseModel):
    """Configuration for browser automation."""

    headless: bool = True
    viewport: Optional[Dict[str, int]] = None
    browser_args: List[str] = []

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True
