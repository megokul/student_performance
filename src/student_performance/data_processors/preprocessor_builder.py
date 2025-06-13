from typing import Optional, Dict, Tuple
from sklearn.pipeline import Pipeline
from box import ConfigBox

from src.student_performance.data_processors.imputer_factory import ImputerFactory
from src.student_performance.data_processors.scaler_factory import ScalerFactory
from src.student_performance.data_processors.encoder_factory import EncoderFactory
from src.student_performance.data_processors.column_math_factory import ColumnMathFactory
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger


class PreprocessorBuilder:
    """
    Dynamically builds X and Y preprocessing pipelines using configurable steps and methods.

    Supported YAML step keys:
        - imputation      → knn, simple, iterative (via ImputerFactory)
        - standardization → standard_scaler, minmax_scaler, robust_scaler (via ScalerFactory)
        - encoding        → one_hot, ordinal (via EncoderFactory)
        - column_math     → mean, add, power, etc. (via ColumnMathFactory)
    """

    @staticmethod
    def _build_column_math(method: str, params: Dict):
        try:
            return ColumnMathFactory(
                columns=params["input_column"],
                operation=method,
                output_column=params["output_column"],
            )
        except KeyError as e:
            raise ValueError(f"Missing required parameter for column_math: {e}") from e

    STEP_BUILDERS = {
        "imputation": ImputerFactory.get_imputer_pipeline,
        "standardization": ScalerFactory.get_scaler_pipeline,
        "encoding": EncoderFactory.get_encoder_pipeline,
        "column_math": _build_column_math.__func__,
    }

    def __init__(self, steps: Optional[Dict] = None, methods: Optional[Dict] = None) -> None:
        """
        Args:
            steps (Dict): Ordered step names per section (e.g., {"x": ["imputation", "encoding"]})
            methods (Dict): Method config per section (e.g., {"x": {"imputation": {...}}})
        """
        self.steps = steps or {}
        self.methods = methods or {}

    def _build_pipeline(self, section: str) -> Pipeline:
        """
        Build a scikit-learn pipeline for a given section ("x" or "y").

        Args:
            section (str): Section name ('x' or 'y')

        Returns:
            Pipeline: A scikit-learn Pipeline object
        """
        try:
            pipeline_steps = []
            step_list = self.steps.get(section, [])
            section_methods = self.methods.get(section, {})

            for step_name in step_list:
                step_config = section_methods.get(step_name, {})

                # Skip if explicitly set to "none"
                if not step_config or (
                    isinstance(step_config, str) and step_config.lower() == "none"
                ):
                    logger.info(
                        f"Skipping '{step_name}' step in section '{section}' as it is set to 'none'."
                    )
                    continue

                builder_fn = self.STEP_BUILDERS.get(step_name)
                if not builder_fn:
                    raise ValueError(f"Unsupported preprocessing step: '{step_name}'")

                if isinstance(step_config, (dict, ConfigBox)):
                    method_name = step_config.get("method")
                    step_params = {k: v for k, v in step_config.items() if k != "method"}
                else:
                    method_name = step_config
                    step_params = {}

                step_object = builder_fn(method_name, step_params)
                pipeline_steps.append((step_name, step_object))

            return Pipeline(pipeline_steps)

        except Exception as e:
            raise StudentPerformanceError(e, logger) from e

    def build(self) -> Tuple[Pipeline, Optional[Pipeline]]:
        """
        Builds preprocessing pipelines for both features (X) and target (Y).

        Returns:
            Tuple[Pipeline, Optional[Pipeline]]: Tuple containing x_pipeline and y_pipeline
        """
        try:
            x_pipeline = self._build_pipeline("x")
            y_pipeline = self._build_pipeline("y")
            return x_pipeline, y_pipeline
        except Exception as e:
            raise StudentPerformanceError(e, logger) from e
