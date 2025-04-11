class TaskExecutionError(Exception):
    """Custom exception for task execution errors."""
    def __init__(self, message: str, original_error: Exception = None):
        self.message = message
        self.original_error = original_error
        super().__init__(self.message)

    def __str__(self):
        if self.original_error:
            return f"{self.message} (Original error: {str(self.original_error)})"
        return self.message 