from typing import Literal
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger

ColumnMathOperation = Literal[
    "add", "subtract", "multiply", "divide",
    "mean_of_columns", "sqrt", "square", "power",
]


class ColumnMathFactory(BaseEstimator, TransformerMixin):
    """
    Transformer to apply mathematical operations to specified columns
    and create a new output column—optionally dropping the inputs in place.

    Attributes:
        columns (list[str]): Columns to operate on.
        operation (str): Operation to apply.
        output_column (str): Name of the resulting column.
        inplace (bool): If True, drops the input columns after transformation.
        return_numpy (bool): If True, returns result as numpy array.
    """

    def __init__(
        self,
        columns: list[str],
        operation: ColumnMathOperation,
        output_column: str,
        inplace: bool = False,
        return_numpy: bool = True,
    ) -> None:
        self.columns = columns
        self.operation = operation.lower()
        self.output_column = output_column
        self.inplace = inplace
        self.return_numpy = return_numpy

        logger.debug(
            "Initialized ColumnMathFactory with operation='%s', columns=%s, output_column='%s', inplace=%s, return_numpy=%s",
            self.operation, self.columns, self.output_column, self.inplace, self.return_numpy
        )

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "ColumnMathFactory":
        logger.debug("Fitting ColumnMathFactory (no operation needed).")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        try:
            logger.debug("Starting transformation using ColumnMathFactory.")
            df = X.copy()

            if self.operation == "add":
                df[self.output_column] = df[self.columns].sum(axis=1)
            elif self.operation == "subtract":
                df[self.output_column] = df[self.columns[0]]
                for col in self.columns[1:]:
                    df[self.output_column] -= df[col]
            elif self.operation == "multiply":
                df[self.output_column] = df[self.columns].prod(axis=1)
            elif self.operation == "divide":
                df[self.output_column] = df[self.columns[0]]
                for col in self.columns[1:]:
                    df[self.output_column] /= df[col]
            elif self.operation == "mean_of_columns":
                df[self.output_column] = df[self.columns].mean(axis=1)
            else:
                logger.error("Unsupported operation '%s' in ColumnMathFactory.", self.operation)
                raise ValueError(f"Unsupported operation: {self.operation}")

            logger.info(
                "Applied column math: operation='%s', columns=%s, output='%s', inplace=%s",
                self.operation,
                self.columns,
                self.output_column,
                self.inplace,
            )

            if self.inplace:
                logger.debug("Dropping source columns: %s", self.columns)
                df.drop(columns=self.columns, inplace=True)

            logger.debug("ColumnMathFactory transformation complete. Output shape: %s", df.shape)

            return df.to_numpy() if self.return_numpy else df

        except Exception as e:
            logger.exception("ColumnMathFactory transformation failed.")
            raise StudentPerformanceError(e, logger) from e
