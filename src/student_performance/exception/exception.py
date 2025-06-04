import sys
from types import TracebackType

from src.student_performance.logging import logger


class StudentPerformanceError(Exception):
    """
    Custom exception for the Student Performance project.

    Automatically captures:
    - Original exception message or optional custom message
    - Filename and line number from traceback
    - Logs the formatted error using a centralized logger
    """

    def __init__(self, error: Exception, message: str | None = None) -> None:
        final_message: str = message or str(error)
        super().__init__(final_message)
        self.message: str = final_message

        # Extract traceback information
        _, _, tb = sys.exc_info()
        tb: TracebackType | None
        self.line: int | None = tb.tb_lineno if tb else None
        self.file: str = tb.tb_frame.f_code.co_filename if tb else "Unknown"

        # Log the error using centralized logger
        logger.error(str(self), exc_info=sys.exc_info())

    def __str__(self) -> str:
        return (
            f"Error occurred in file [{self.file}], "
            f"line [{self.line}], "
            f"message: [{self.message}]"
        )
