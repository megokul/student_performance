from typing import Optional
from ensure import ensure_annotations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger


class ScalerFactory:
    """
    Factory to build scaling pipelines for numerical features.
    """

    _SUPPORTED_METHODS = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }

    @staticmethod
    @ensure_annotations
    def get_scaler_pipeline(
        method: str,
        params: Optional[dict] = None,
        is_target: bool = False
    ) -> Pipeline:
        try:
            if method not in ScalerFactory._SUPPORTED_METHODS:
                raise ValueError(f"Unsupported scaler method: {method}")

            scaler_cls = ScalerFactory._SUPPORTED_METHODS[method]
            scaler = scaler_cls(**(params or {}))

            return Pipeline(steps=[("scaler", scaler)])

        except Exception as e:
            raise StudentPerformanceError(e, logger) from e
