from typing import Dict, Tuple
from box import ConfigBox
from sklearn.pipeline import Pipeline

from src.student_performance.data_processors.imputer_factory import ImputerFactory
from src.student_performance.data_processors.scaler_factory import ScalerFactory
from src.student_performance.data_processors.encoder_factory import EncoderFactory
from src.student_performance.data_processors.column_math_factory import ColumnMathFactory
from src.student_performance.data_processors.column_operation_factory import get_column_operation
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger


class PreprocessorBuilder:
    """
    Dynamically builds X and Y preprocessing pipelines using configurable
    steps and methods from the transformation config.
    """

    @staticmethod
    def _build_column_math(method: str, params: Dict[str, object]) -> ColumnMathFactory:
        try:
            logger.debug("Building 'column_math' step with method='%s', params=%s", method, params)
            input_cols = params["input_column"]
            output_col = params["output_column"]
            inplace = params.get("inplace", False)

            return ColumnMathFactory(
                columns=input_cols,
                operation=method,
                output_column=output_col,
                inplace=inplace,
            )
        except KeyError as e:
            logger.error("Missing required parameter in 'column_math': %s", e)
            raise ValueError(f"Missing required parameter for column_math: {e}") from e
        except Exception as e:
            logger.exception("Failed to build 'column_math' step.")
            raise StudentPerformanceError(e, logger) from e

    @staticmethod
    def _build_column_operation(method: str, params: Dict[str, object]):
        try:
            logger.debug("Building 'column_operation' step with method='%s', params=%s", method, params)
            return get_column_operation(method, **params)
        except Exception as e:
            logger.exception("Failed to build 'column_operation' step.")
            raise StudentPerformanceError(e, logger) from e

    STEP_BUILDERS: dict[str, object] = {
        "column_operation": _build_column_operation.__func__,
        "imputation": ImputerFactory.get_imputer_pipeline,
        "standardization": ScalerFactory.get_scaler_pipeline,
        "encoding": EncoderFactory.get_encoder_pipeline,
        "column_math": _build_column_math.__func__,
    }

    def __init__(
        self,
        steps: dict[str, list[str]] | None = None,
        methods: dict[str, ConfigBox] | None = None
    ) -> None:
        self.steps = steps or {}
        self.methods = methods or {}

    def _build_pipeline(self, section: str) -> Pipeline:
        try:
            logger.info("Building preprocessing pipeline for section: '%s'", section)

            pipeline_steps: list[tuple[str, object]] = []
            step_list = self.steps.get(section, [])
            section_methods = self.methods.get(section, {})

            for step_name in step_list:
                step_config = section_methods.get(step_name, {})

                if not step_config or (
                    isinstance(step_config, str)
                    and step_config.lower() == "none"
                ):
                    logger.info("Skipping step '%s' in section '%s' (explicitly disabled)", step_name, section)
                    continue

                builder_fn = self.STEP_BUILDERS.get(step_name)
                if builder_fn is None:
                    logger.error("Unsupported preprocessing step '%s' in section '%s'", step_name, section)
                    raise ValueError(f"Unsupported preprocessing step: '{step_name}'")

                # Extract method name and parameters
                if isinstance(step_config, (dict, ConfigBox)):
                    method_name = step_config.get("method")
                    step_params = {k: v for k, v in step_config.items() if k != "method"}
                else:
                    method_name = step_config
                    step_params = {}

                logger.debug("Building step '%s' with method='%s' and params=%s", step_name, method_name, step_params)
                step_obj = builder_fn(method_name, step_params)

                logger.info("Adding step '%s' to section '%s'", step_name, section)
                pipeline_steps.append((step_name, step_obj))

            logger.info("Completed pipeline for section '%s' with %d steps", section, len(pipeline_steps))
            return Pipeline(pipeline_steps)

        except Exception as e:
            logger.exception("Failed to build preprocessing pipeline for section: '%s'", section)
            raise StudentPerformanceError(e, logger) from e

    def build(self) -> Tuple[Pipeline, Pipeline | None]:
        """
        Builds and returns the X and Y preprocessing pipelines.
        """
        try:
            logger.info("Initiating full preprocessing pipeline build (X and Y).")
            x_pipeline = self._build_pipeline("x")
            y_pipeline = self._build_pipeline("y")
            logger.info("Successfully built both X and Y preprocessing pipelines.")
            return x_pipeline, y_pipeline
        except Exception as e:
            logger.exception("Pipeline build failed during construction of X and Y pipelines.")
            raise StudentPerformanceError(e, logger) from e
