class ControlledDecodeError(Exception):
    """Expected per-sample decode failure that should be logged without traceback."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason

