from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

from src.student_performance.logging import logger
from src.student_performance.exception.exception import StudentPerformanceError


class ColumnOperationFactory(BaseEstimator, TransformerMixin):
    """
    Factory transformer for performing column-level operations.

    Supported operations:
        - remove_col: Removes specified columns from the dataset
    """

    def __init__(self, operation: str, columns: list[str] | None = None):
        """
        Args:
            operation (str): The operation to apply (e.g., "remove_col").
            columns (list[str]): Columns involved in the operation.
        """
        self.operation = operation.lower()
        self.columns = columns or []

    def fit(self, X: pd.DataFrame, y=None) -> "ColumnOperationFactory":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame.")

            logger.debug("Applying column operation: '%s' on columns: %s", self.operation, self.columns)

            if self.operation == "remove_col":
                transformed = X.drop(columns=self.columns, errors="ignore")
                logger.info("Removed columns: %s (if present)", self.columns)
                return transformed

            raise ValueError(f"Unsupported column operation: '{self.operation}'")

        except Exception as e:
            logger.exception("Failed to perform column operation: %s", self.operation)
            raise StudentPerformanceError(e, logger) from e


def get_column_operation(method: str, **kwargs) -> ColumnOperationFactory:
    """
    Factory method to return a configured ColumnOperationFactory instance.

    Supported method:
        - 'remove_col'

    Example:
        get_column_operation("remove_col", columns=["id", "timestamp"])
    """
    try:
        return ColumnOperationFactory(operation=method, columns=kwargs.get("columns", []))
    except Exception as e:
        logger.exception("Failed to initialize ColumnOperationFactory for method: %s", method)
        raise StudentPerformanceError(e, logger) from e
