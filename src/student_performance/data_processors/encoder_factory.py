# FILE: src/student_performance/data_processors/encoder_factory.py

from typing import Optional
from ensure import ensure_annotations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import pandas as pd

from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger

from sklearn.compose import ColumnTransformer

class EncoderFactory:
    _SUPPORTED_METHODS = {
        "one_hot": OneHotEncoder,
        "ordinal": OrdinalEncoder,
    }

    @staticmethod
    @ensure_annotations
    def get_encoder_pipeline(
        method: str,
        params: Optional[dict] = None,
        is_target: bool = False,
    ) -> Pipeline:
        try:
            if method not in EncoderFactory._SUPPORTED_METHODS:
                raise ValueError(f"Unsupported encoding method: {method}")

            encoder_class = EncoderFactory._SUPPORTED_METHODS[method]

            # Extract and remove 'columns' from params
            columns = params.pop("columns", None)
            if columns is None:
                raise ValueError("You must specify 'columns' for encoder in params.yaml")

            encoder = encoder_class(**params)

            # Wrap with ColumnTransformer
            column_transformer = ColumnTransformer(
                transformers=[("encoder", encoder, columns)],
                remainder="passthrough"
            )

            return Pipeline(steps=[("column_encoder", column_transformer)])

        except Exception as e:
            raise StudentPerformanceError(e, logger) from e

