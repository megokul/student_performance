from typing import Optional
from ensure import ensure_annotations
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger


class ImputerFactory:
    """
    Factory to build imputation pipelines for numerical data.

    Supported methods:
        - knn
        - simple
        - iterative
        - custom (requires callable in params)
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
            logger.debug("Requested imputer: method='%s', is_target=%s", method, is_target)

            params = params or {}

            if method == "custom":
                if "custom_callable" not in params:
                    raise ValueError("Custom imputer requires a 'custom_callable' in params.")
                logger.info("Using custom imputer callable.")
                imputer = params["custom_callable"]()
            else:
                ImputerClass = ImputerFactory._SUPPORTED_METHODS.get(method)
                if not ImputerClass:
                    raise ValueError(f"Unsupported imputation method: {method}")

                logger.info("Initializing %s imputer with params: %s", method, params)
                imputer = ImputerClass(**params)

            logger.info("Successfully created imputer pipeline using method: %s", method)
            return Pipeline(steps=[("imputer", imputer)])

        except Exception as e:
            logger.exception("Failed to build imputer pipeline using method: %s", method)
            raise StudentPerformanceError(e, logger) from e
