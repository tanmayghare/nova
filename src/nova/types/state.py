from typing import Any, Dict, Optional

from pydantic import BaseModel


class State(BaseModel):
    """Represents the current state of the browser."""

    url: str
    html: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = {}

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True
