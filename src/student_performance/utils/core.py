import pandas as pd
from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError, BoxTypeError, BoxKeyError
from ensure import ensure_annotations
import yaml

from src.student_performance.logging import logger
from src.student_performance.exception.exception import StudentPerformanceError


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Load a YAML file and return its contents as a ConfigBox for dot-access.

    Raises:
        StudentPerformanceError: If the file is missing, corrupted, or unreadable.
    """
    if not path_to_yaml.exists():
        msg = f"YAML file not found: '{path_to_yaml}'"
        raise StudentPerformanceError(FileNotFoundError(msg), msg)

    try:
        with path_to_yaml.open("r", encoding="utf-8") as file:
            content = yaml.safe_load(file)
    except (BoxValueError, BoxTypeError, BoxKeyError, yaml.YAMLError) as e:
        msg = f"Failed to parse YAML from: '{path_to_yaml.as_posix()}' — {e}"
        raise StudentPerformanceError(e, msg) from e
    except Exception as e:
        msg = f"Unexpected error while reading YAML from: '{path_to_yaml.as_posix()}' — {e}"
        raise StudentPerformanceError(e, msg) from e

    if content is None:
        msg = f"YAML file is empty or improperly formatted: '{path_to_yaml}'"
        raise StudentPerformanceError(ValueError(msg), msg)

    logger.info(f"YAML successfully loaded from: '{path_to_yaml.as_posix()}'")
    return ConfigBox(content)


@ensure_annotations
def save_to_csv(df: pd.DataFrame, *paths: Path, label: str):
    try:
        for path in paths:
            path = Path(path)
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory for {label}: '{path.parent.as_posix()}'")
            else:
                logger.info(f"Directory already exists for {label}: '{path.parent.as_posix()}'")

            df.to_csv(path, index=False)
            logger.info(f"{label} saved to: '{path.as_posix()}'")
    except Exception as e:
        msg = f"Failed to save CSV to: '{path.as_posix()}' — {e}"
        raise StudentPerformanceError(msg, logger) from e


@ensure_annotations
def read_csv(filepath: Path) -> pd.DataFrame:
    """
    Read a CSV file into a Pandas DataFrame.

    Raises:
        StudentPerformanceError: If the file is missing, corrupted, or unreadable.
    """
    if not filepath.exists():
        msg = f"CSV file not found: '{filepath}'"
        raise StudentPerformanceError(FileNotFoundError(msg), msg)

    try:
        df = pd.read_csv(filepath)
        logger.info(f"CSV file read successfully from: '{filepath.as_posix()}'")
        return df
    except Exception as e:
        msg = f"Failed to read CSV from: '{filepath.as_posix()}' — {e}"
        raise StudentPerformanceError(e, msg) from e
