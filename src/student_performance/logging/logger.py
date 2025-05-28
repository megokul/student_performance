import logging
import sys
from pathlib import Path
from src.student_performance.constants.constants import LOGS_ROOT
from src.student_performance.utils.timestamp import get_utc_timestamp

def setup_logger(name: str = "app_logger", level: int = logging.DEBUG) -> logging.Logger:
    """
    Set up and return a logger with file and stream handlers, allowing custom log level.

    Args:
        name (str): Name of the logger.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    sys.stdout.reconfigure(encoding="utf-8")
    timestamp = get_utc_timestamp()

    log_dir = Path(LOGS_ROOT) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filepath = log_dir / f"{timestamp}.log"

    log_format = "[%(asctime)s] - %(levelname)s - %(module)s - %(message)s"
    formatter = logging.Formatter(log_format)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Add file handler if not already added
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_filepath)
               for h in logger.handlers):
        file_handler = logging.FileHandler(log_filepath, mode="a")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    # Add stdout stream handler if not already added
    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
               for h in logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    return logger
