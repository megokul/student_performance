import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from src.student_performance.logging import logger
from src.student_performance.exception.exception import StudentPerformanceError


@dataclass(frozen=True)
class StudentPerformanceModel:
    """
    Wraps a trained model plus its X/Y preprocessors for inference.
    """
    model: Any
    x_preprocessor: Any | None = None
    # y_preprocessor: Any | None = None

    def predict(self, X):
        """
        Apply the X-preprocessor (if any), run model.predict, then
        inverse-transform via the Y-preprocessor (if any).
        """
        try:
            data = X
            if self.x_preprocessor is not None:
                data = self.x_preprocessor.transform(data)

            preds = self.model.predict(data)

            # if self.y_preprocessor is not None:
            #     preds = self.y_preprocessor.inverse_transform(preds)

            return preds

        except Exception as e:
            logger.exception("Failed during model prediction.")
            raise StudentPerformanceError(e, logger) from e

    @classmethod
    def from_artifacts(
        cls,
        model_path: Union[Path, str],
        x_preprocessor_path: Union[Path, str] | None = None,
        # y_preprocessor_path: Union[Path, str] | None = None,
    ):
        """
        Load model and optional preprocessors from disk (joblib .joblib files).
        """
        try:
            model = joblib.load(model_path)

            x_proc = (
                joblib.load(x_preprocessor_path)
                if x_preprocessor_path
                else None
            )
            # y_proc = (
            #     joblib.load(y_preprocessor_path)
            #     if y_preprocessor_path
            #     else None
            # )

            return cls(
                model=model,
                x_preprocessor=x_proc,
                # y_preprocessor=y_proc,
            )

        except Exception as e:
            logger.exception("Failed to load artifacts for inference model.")
            raise StudentPerformanceError(e, logger) from e

    @classmethod
    def from_objects(
        cls,
        model: Any,
        x_preprocessor: Any | None = None,
        # y_preprocessor: Any | None = None,
    ):
        """
        Construct directly from in-memory model and preprocessor objects.
        """
        try:
            return cls(
                model=model,
                x_preprocessor=x_preprocessor,
                # y_preprocessor=y_preprocessor,
            )
        except Exception as e:
            logger.exception("Failed to build inference model from objects.")
            raise StudentPerformanceError(e, logger) from e
