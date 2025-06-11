from typing import Literal
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ensure import ensure_annotations

from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger


ColumnMathOperation = Literal[
    "add", "subtract", "multiply", "divide",
    "mean", "sqrt", "square", "power"
]


class ColumnMathFactory(BaseEstimator, TransformerMixin):
    """
    Transformer to apply mathematical operations to specified columns
    and create a new output column.
    """

    @ensure_annotations
    def __init__(
        self,
        columns: list[str],
        operation: ColumnMathOperation,
        output_column: str
    ) -> None:
        self.columns = columns
        self.operation = operation.lower()
        self.output_column = output_column

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> "ColumnMathFactory":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
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

            elif self.operation == "mean":
                df[self.output_column] = df[self.columns].mean(axis=1)

            elif self.operation == "sqrt":
                if len(self.columns) != 1:
                    raise ValueError("sqrt operation requires exactly 1 column.")
                df[self.output_column] = np.sqrt(df[self.columns[0]])

            elif self.operation == "square":
                if len(self.columns) != 1:
                    raise ValueError("square operation requires exactly 1 column.")
                df[self.output_column] = df[self.columns[0]] ** 2

            elif self.operation == "power":
                if len(self.columns) != 2:
                    raise ValueError("power operation requires exactly 2 columns.")
                df[self.output_column] = df[self.columns[0]] ** df[self.columns[1]]

            else:
                raise ValueError(f"Unsupported operation: {self.operation}")

            logger.info(
                "Applied '%s' operation on columns: %s → '%s'",
                self.operation,
                self.columns,
                self.output_column
            )

            return df

        except Exception as e:
            raise StudentPerformanceError(e, logger) from e
