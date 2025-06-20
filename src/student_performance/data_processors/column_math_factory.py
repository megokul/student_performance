from typing import Literal
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from ensure import ensure_annotations

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
    """

    def __init__(
        self,
        columns: list[str],
        operation: ColumnMathOperation,
        output_column: str,
        inplace: bool = False,
    ) -> None:
        self.columns = columns
        self.operation = operation.lower()
        self.output_column = output_column
        self.inplace = inplace

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

            elif self.operation == "mean_of_columns":
                df[self.output_column] = df[self.columns].mean(axis=1)

            else:
                raise ValueError(f"Unsupported operation: {self.operation}")

            logger.info(
                "Applied '%s' on %s → '%s' (inplace=%s)",
                self.operation,
                self.columns,
                self.output_column,
                self.inplace,
            )

            if self.inplace:
                df.drop(columns=self.columns, inplace=True)

            return df

        except Exception as e:
            raise StudentPerformanceError(e, logger) from e
