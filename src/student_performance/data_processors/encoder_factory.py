# FILE: src/student_performance/data_processors/encoder_factory.py

from typing import Optional
from ensure import ensure_annotations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

import pandas as pd

from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger


class EncoderFactory:
    """
    Factory to construct encoding pipelines.

    Supported methods:
        - one_hot   → OneHotEncoder
        - ordinal   → OrdinalEncoder
    """
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
            logger.debug("Requested encoder: method='%s', is_target=%s", method, is_target)

            if method not in EncoderFactory._SUPPORTED_METHODS:
                raise ValueError(f"Unsupported encoding method: {method}")

            encoder_class = EncoderFactory._SUPPORTED_METHODS[method]
            params = params or {}

            # Extract and remove 'columns' from params
            columns = params.pop("columns", None)
            if columns is None:
                raise ValueError("You must specify 'columns' for encoder in params.yaml")

            logger.info("Initializing %s with columns: %s and params: %s", method, columns, params)

            encoder = encoder_class(**params)

            column_transformer = ColumnTransformer(
                transformers=[("encoder", encoder, columns)],
                remainder="passthrough"
            )

            logger.info("Successfully built ColumnTransformer with method: %s", method)

            return Pipeline(steps=[("column_encoder", column_transformer)])

        except Exception as e:
            logger.exception("Failed to build encoder pipeline using method: %s", method)
            raise StudentPerformanceError(e, logger) from e
