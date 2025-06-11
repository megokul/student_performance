from typing import Tuple
from ensure import ensure_annotations
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  # noqa

from src.student_performance.data_processors.imputer_factory import ImputerFactory
from src.student_performance.data_processors.scaler_factory import ScalerFactory
from src.student_performance.data_processors.encoder_factory import EncoderFactory
from src.student_performance.data_processors.column_math_factory import ColumnMathFactory
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger


class PreprocessorBuilder:
    """
    Builds preprocessing pipelines for features (X) and target (y).
    """

    STEP_BUILDERS = {
        "imputation": ImputerFactory.get_imputer_pipeline,
        "encoding": EncoderFactory.get_encoder_pipeline,
        "standardization": ScalerFactory.get_scaler_pipeline,
        "compute_target": lambda method, params: ColumnMathTransformer(
            columns=params.get("input_column", []),
            output_column=params.get("output_column", "target"),
            operation="mean"
        ),
    }

    @ensure_annotations
    def __init__(self, steps: dict, methods: dict) -> None:
        self.steps = steps or {}
        self.methods = methods or {}

    def _build_pipeline(self, section: str) -> Pipeline:
        try:
            pipeline_steps = []
            ordered_steps = self.steps.get(section, [])
            method_map = self.methods.get(section, {})

            for step_name in ordered_steps:
                params = method_map.get(step_name, {})
                method = params.get("method")

                if not method or str(method).lower() == "none":
                    logger.info(f"Skipping '{step_name}' for '{section}' (disabled).")
                    continue

                builder = self.STEP_BUILDERS.get(step_name)
                if not builder:
                    raise ValueError(f"Unsupported step '{step_name}' in '{section}' pipeline.")

                component = builder(method, params)
                pipeline_steps.append((step_name, component))

            return Pipeline(steps=pipeline_steps)

        except Exception as e:
            raise StudentPerformanceError(e, logger) from e

    @ensure_annotations
    def build(self) -> Tuple[Pipeline, Pipeline]:
        try:
            x_pipeline = self._build_pipeline("x")
            y_pipeline = self._build_pipeline("y")
            return x_pipeline, y_pipeline
        except Exception as e:
            raise StudentPerformanceError(e, logger) from e
