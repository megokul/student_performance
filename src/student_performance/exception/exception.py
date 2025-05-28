import sys
from typing import Protocol
from types import TracebackType


class LoggerInterface(Protocol):
    def error(self, message: str) -> None:
        """Log an error-level message."""
        ...


class StudentPerformanceError(Exception):
    def __init__(self, error: Exception, logger: LoggerInterface) -> None:
        """
        Wraps and logs the original exception with traceback metadata.
        """
        super().__init__(str(error))
        self.message: str = str(error)

        # Extract traceback info
        _, _, tb: TracebackType | None = sys.exc_info()
        self.line: int | None = tb.tb_lineno if tb else None
        self.file: str = tb.tb_frame.f_code.co_filename if tb else "Unknown"

        # Log the error using the provided logger
        logger.error(str(self))

    def __str__(self) -> str:
        return (
            f"Error occurred in file [{self.file}], "
            f"line [{self.line}], "
            f"message: [{self.message}]"
        )
