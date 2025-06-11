from typing import Optional
from ensure import ensure_annotations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger


class EncoderFactory:
    """
    Factory to build encoding pipelines for categorical features.
    """

    _SUPPORTED_METHODS = {
        "onehot": OneHotEncoder,
        "ordinal": OrdinalEncoder,
    }

    @staticmethod
    @ensure_annotations
    def get_encoder_pipeline(
        method: str,
        params: Optional[dict] = None,
        is_target: bool = False
    ) -> Pipeline:
        try:
            if method not in EncoderFactory._SUPPORTED_METHODS:
                raise ValueError(f"Unsupported encoding method: {method}")

            encoder_class = EncoderFactory._SUPPORTED_METHODS[method]
            encoder = encoder_class(**(params or {}))
            return Pipeline(steps=[("encoder", encoder)])

        except Exception as e:
            raise StudentPerformanceError(e, logger) from e
