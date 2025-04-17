import os
from typing import List

from pydantic import BaseModel, ConfigDict


class BrowserConfig(BaseModel):
    """Configuration for browser automation."""

    headless: bool = os.environ.get("BROWSER_HEADLESS", "True").lower() == "true"
    viewport_width: int = os.environ.get("BROWSER_VIEWPORT_WIDTH", "1280")
    viewport_height: int = os.environ.get("BROWSER_VIEWPORT_HEIGHT", "800")
    browser_args: List[str] = []
    action_timeout: int = int(os.environ.get("BROWSER_ACTION_TIMEOUT", "15"))
    extra_args: List[str] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)
