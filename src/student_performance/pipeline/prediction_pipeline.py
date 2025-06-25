from pathlib import Path
import numpy as np

from src.student_performance.config.configuration import ConfigurationManager
from src.student_performance.dbhandler.s3_handler import S3Handler
from src.student_performance.components.model_prediction import ModelPrediction
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger
from src.student_performance.utils.core import load_array


class PredictionPipeline:
    def __init__(self):
        try:
            logger.info("Initializing PredictionPipeline...")
            self.config_manager = ConfigurationManager()
        except Exception as e:
            raise StudentPerformanceError(e, "Failed to initialize PredictionPipeline.") from e

    def run_pipeline(self, input_data_file: Path | None = None, input_array: np.ndarray | None = None) -> np.ndarray:
        try:
            logger.info("========== Prediction Pipeline Started ==========")

            # Step 1: Load prediction configuration and backup handler
            prediction_config = self.config_manager.get_model_prediction_config()
            s3_config = self.config_manager.get_s3_handler_config()
            s3_handler = S3Handler(s3_config) if prediction_config.s3_enabled else None

            # Step 2: Initialize the model predictor
            predictor = ModelPrediction(
                prediction_config=prediction_config,
                backup_handler=s3_handler,
            )

            # Step 3: Prepare input data
            if input_array is not None:
                logger.info("Using directly provided input array for prediction.")
                X = input_array
            elif input_data_file is not None:
                logger.info(f"Loading input data from file: {input_data_file}")
                X = load_array(input_data_file, label="Input Data")
            else:
                raise StudentPerformanceError("No input data provided for prediction.", logger)

            # Step 4: Run prediction
            predictions = predictor.predict(X)

            # Step 5: Save predictions to local/S3
            predictor.save_predictions(predictions)

            logger.info("========== Prediction Pipeline Completed ==========")
            return predictions

        except Exception as e:
            raise StudentPerformanceError(e, "PredictionPipeline failed.") from e
