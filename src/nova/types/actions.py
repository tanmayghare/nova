from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class Action(BaseModel):
    """Base class for actions."""

    type: str
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary.

        Returns:
            Dictionary representation of the action
        """
        return {"type": self.type, "parameters": self.parameters}


class ActionResult(BaseModel):
    """Result of an action execution."""

    is_success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

    @classmethod
    def success(
        cls, result: Any = None, metadata: Optional[Dict[str, Any]] = None
    ) -> ActionResult:
        """Create a successful result.

        Args:
            result: Result value
            metadata: Additional metadata

        Returns:
            Successful ActionResult
        """
        return cls(is_success=True, result=result, metadata=metadata or {})

    @classmethod
    def failure(
        cls, error: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ActionResult:
        """Create a failed result.

        Args:
            error: Error message
            metadata: Additional metadata

        Returns:
            Failed ActionResult
        """
        return cls(is_success=False, error=error, metadata=metadata or {})
