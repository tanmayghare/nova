"""Custom exceptions for Nova."""

class NovaError(Exception):
    """Base exception for all Nova errors."""
    def __init__(self, message: str, original_error: Exception = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)

    def __str__(self):
        if self.original_error:
            return f"{self.message} (Original error: {str(self.original_error)})"
        return self.message


class AgentError(NovaError):
    """Exception for agent-related errors."""
    pass


class LLMError(NovaError):
    """Exception for LLM-related errors."""
    pass


class MaxStepsExceededError(AgentError):
    """Exception for when an agent exceeds its maximum allowed steps."""
    pass


class TaskExecutionError(NovaError):
    """Exception for task execution errors."""
    pass 