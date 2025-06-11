from typing import Optional
from ensure import ensure_annotations
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer

from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger


class ImputerFactory:
    """
    Factory to build imputation pipelines for numerical data.
    """

    _SUPPORTED_METHODS = {
        "knn": KNNImputer,
        "simple": SimpleImputer,
        "iterative": IterativeImputer,
    }

    @staticmethod
    @ensure_annotations
    def get_imputer_pipeline(
        method: str,
        params: Optional[dict] = None,
        is_target: bool = False
    ) -> Pipeline:
        try:
            if method == "custom":
                if not params or "custom_callable" not in params:
                    raise ValueError("Custom imputer requires a 'custom_callable' in params.")
                imputer = params["custom_callable"]()
            else:
                ImputerClass = ImputerFactory._SUPPORTED_METHODS.get(method)
                if not ImputerClass:
                    raise ValueError(f"Unsupported imputation method: {method}")
                imputer = ImputerClass(**(params or {}))

            return Pipeline(steps=[("imputer", imputer)])

        except Exception as e:
            raise StudentPerformanceError(e, logger) from e
