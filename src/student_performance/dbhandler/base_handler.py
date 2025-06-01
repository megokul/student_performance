from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger


class DBHandler(ABC):
    """
    Abstract base class for all database/storage handlers.
    Enables unified behavior across PostgreSQL, MongoDB, CSV, etc.
    """

    def __enter__(self) -> "DBHandler":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self.close()
        except Exception as e:
            msg = "Error closing DBHandler."
            raise StudentPerformanceError(e, msg) from e

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def load_from_source(self) -> pd.DataFrame:
        pass

    def load_from_csv(self, source: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(source)
            logger.info(f"DataFrame loaded from CSV: {source}")
            return df
        except Exception as e:
            msg = f"Failed to load DataFrame from CSV: '{source}'"
            raise StudentPerformanceError(e, msg) from e
