from __future__ import annotations

from typing import Dict, Tuple

from box import ConfigBox
from sklearn.pipeline import Pipeline

from src.student_performance.data_processors.imputer_factory import ImputerFactory
from src.student_performance.data_processors.scaler_factory import ScalerFactory
from src.student_performance.data_processors.encoder_factory import EncoderFactory
from src.student_performance.data_processors.column_math_factory import ColumnMathFactory
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger


class PreprocessorBuilder:
    """
    Dynamically builds X and Y preprocessing pipelines using configurable
    steps and methods.

    Supported YAML step keys:
      - imputation      → knn, simple, iterative (via ImputerFactory)
      - standardization → standard_scaler, minmax_scaler, robust_scaler
      - encoding        → one_hot, ordinal (via EncoderFactory)
      - column_math     → mean, add, power, etc. (via ColumnMathFactory)
    """

    @staticmethod
    def _build_column_math(method: str, params: Dict[str, object]) -> ColumnMathFactory:
        try:
            input_cols = params["input_column"] 
            output_col = params["output_column"]
            inplace = params.get("inplace", False)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for column_math: {e}") from e

        return ColumnMathFactory(
            columns=input_cols,  # type: ignore[arg-type]
            operation=method,
            output_column=output_col,
            inplace=inplace,
        )

    STEP_BUILDERS: dict[str, object] = {
        "imputation": ImputerFactory.get_imputer_pipeline,
        "standardization": ScalerFactory.get_scaler_pipeline,
        "encoding": EncoderFactory.get_encoder_pipeline,
        "column_math": _build_column_math.__func__,
    }

    def __init__(self, steps: dict[str, list[str]] | None = None, methods: dict[str, ConfigBox] | None = None) -> None:
        """
        Args:
            steps:   Ordered step names per section
                     (e.g., {"x": ["imputation", "encoding"]})
            methods: Method config per section
                     (e.g., {"x": {"imputation": {...}}})
        """
        self.steps = steps or {}
        self.methods = methods or {}

    def _build_pipeline(self, section: str) -> Pipeline:
        """
        Build a scikit-learn pipeline for a given section ("x" or "y").

        Args:
            section: Section name ('x' or 'y').

        Returns:
            A scikit-learn Pipeline object.
        """
        try:
            pipeline_steps: list[tuple[str, object]] = []
            step_list = self.steps.get(section, [])
            section_methods = self.methods.get(section, {})

            for step_name in step_list:
                step_config = section_methods.get(step_name, {})

                # Skip if explicitly set to "none"
                if not step_config or (
                    isinstance(step_config, str)
                    and step_config.lower() == "none"
                ):
                    logger.info(
                        "Skipping '%s' in '%s' (set to 'none').",
                        step_name,
                        section,
                    )
                    continue

                builder_fn = self.STEP_BUILDERS.get(step_name)
                if builder_fn is None:
                    raise ValueError(
                        f"Unsupported preprocessing step: '{step_name}'"
                    )

                if isinstance(step_config, (dict, ConfigBox)):
                    method_name = step_config.get("method")
                    step_params = {
                        k: v for k, v in step_config.items() if k != "method"
                    }
                else:
                    method_name = step_config
                    step_params = {}

                step_obj = builder_fn(method_name, step_params)
                pipeline_steps.append((step_name, step_obj))

            return Pipeline(pipeline_steps)
        except Exception as e:
            raise StudentPerformanceError(e, logger) from e

    def build(self) -> Tuple[Pipeline, Pipeline | None]:
        """
        Builds preprocessing pipelines for both features (X) and target (Y).

        Returns:
            A tuple: (x_pipeline, y_pipeline or None).
        """
        try:
            x_pipeline = self._build_pipeline("x")
            y_pipeline = self._build_pipeline("y")
            return x_pipeline, y_pipeline
        except Exception as e:
            raise StudentPerformanceError(e, logger) from e
