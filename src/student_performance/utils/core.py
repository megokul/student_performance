import pandas as pd
from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError, BoxTypeError, BoxKeyError
from ensure import ensure_annotations
import yaml
import json
import numpy as np
import pandas as pd
import joblib

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
        raise StudentPerformanceError(e, msg) from e


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

@ensure_annotations
def save_to_yaml(data: dict, *paths: Path, label: str):
    """
    Write a dict out to YAML, always using UTF-8.
    """
    try:
        for path in paths:
            path = Path(path)
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory for {label}: '{path.parent.as_posix()}'")
            else:
                logger.info(f"Directory already exists for {label}: '{path.parent.as_posix()}'")

            # Write UTF-8
            with open(path, "w", encoding="utf-8") as file:
                yaml.dump(data, file, sort_keys=False)

            logger.info(f"{label} saved to: '{path.as_posix()}'")
    except Exception as e:
        msg = f"Failed to read CSV from: '{path.as_posix()}' — {e}"
        raise StudentPerformanceError(e, msg) from e

@ensure_annotations
def save_to_json(data: dict, *paths: Path, label: str):
    try:
        for path in paths:
            path = Path(path)
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory for {label}: '{path.parent.as_posix()}'")
            else:
                logger.info(f"Directory already exists for {label}: '{path.parent.as_posix()}'")

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            logger.info(f"{label} saved to: '{path.as_posix()}'")
    except Exception as e:
        msg = f"Failed to read CSV from: '{path.as_posix()}' — {e}"
        raise StudentPerformanceError(e, msg) from e


@ensure_annotations
def save_object(obj: object, *paths: Path, label: str):
    """
    Saves a serializable object using joblib to the specified path.

    Args:
        obj (object): The object to serialize.
        path (Path): The path to save the object.
        label (str): Label used for logging context.
    """
    try:
        for path in paths:
            path = Path(path)
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory for {label}: '{path.parent.as_posix()}'")
            else:
                logger.info(f"Directory already exists for {label}: '{path.parent.as_posix()}'")

            joblib.dump(obj, path)
            logger.info(f"{label} saved to: '{path.as_posix()}'")

    except Exception as e:
        msg = f"Failed to save {label} to: '{path.as_posix()}'"
        raise StudentPerformanceError(e, logger) from e


@ensure_annotations
def save_array(array: np.ndarray | pd.Series, *paths: Path, label: str):
    """
    Saves a NumPy array or pandas Series to the specified paths in `.npy` format.

    Args:
        array (Union[np.ndarray, pd.Series]): Data to save.
        *paths (Path): One or more file paths.
        label (str): Label for logging.
    """
    try:
        array = np.asarray(array)

        for path in paths:
            path = Path(path)

            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory for {label}: '{path.parent.as_posix()}'")
            else:
                logger.info(f"Directory already exists for {label}: '{path.parent.as_posix()}'")

            np.save(path, array)
            logger.info(f"{label} saved to: '{path.as_posix()}'")

    except Exception as e:
        msg = f"Failed to save {label} to: '{path.as_posix()}'"
        raise StudentPerformanceError(e, logger) from e


@ensure_annotations
def load_array(path: Path, label: str) -> np.ndarray:
    """
    Loads a NumPy array from the specified `.npy` file path.

    Args:
        path (Path): Path to the `.npy` file.
        label (str): Label for logging.

    Returns:
        np.ndarray: Loaded NumPy array.
    """
    try:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"{label} file not found at path: '{path.as_posix()}'")

        array = np.load(path)
        logger.info(f"{label} loaded successfully from: '{path.as_posix()}'")
        return array

    except Exception as e:
        msg = f"Failed to load {label} from: '{path.as_posix()}'"
        raise StudentPerformanceError(e, logger) from e

@ensure_annotations
def load_object(path: Path, label: str):
    """
    Loads a serialized object from the specified path using joblib.

    Args:
        path (Path): The path to the serialized object.
        label (str): Label used for logging context.

    Returns:
        Any: The deserialized object.
    """
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{label} not found at: '{path.as_posix()}'")
        obj = joblib.load(path)
        logger.info(f"{label} loaded from: '{path.as_posix()}'")
        return obj

    except Exception as e:
        msg = f"Failed to load {label} from: '{path.as_posix()}'"
        logger.exception(msg)
        raise StudentPerformanceError(e, logger) from e