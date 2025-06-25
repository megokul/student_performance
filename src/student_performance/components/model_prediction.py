from typing import Any
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from src.student_performance.entity.config_entity import ModelPredictionConfig
from src.student_performance.dbhandler.base_handler import DBHandler
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger
from src.student_performance.utils.core import load_object, load_array
from src.student_performance.inference.estimator import StudentPerformanceModel


class ModelPrediction:
    def __init__(
        self,
        prediction_config: ModelPredictionConfig,
        backup_handler: DBHandler | None = None,
    ):
        try:
            logger.info("Initializing ModelPrediction.")
            self.prediction_config = prediction_config
            self.backup_handler = backup_handler
            self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

            self._load_inference_model()
        except Exception as e:
            logger.exception("Failed to initialize ModelPrediction.")
            raise StudentPerformanceError(e, logger) from e

    def _load_inference_model(self):
        try:
            if self.prediction_config.local_enabled:
                logger.info("Loading inference model from local path.")
                self.inference_model: StudentPerformanceModel = load_object(
                    self.prediction_config.inference_model_filepath,
                    label="Inference Model",
                )
            elif self.prediction_config.s3_enabled and self.backup_handler:
                logger.info("Loading inference model from S3.")
                with self.backup_handler as handler:
                    self.inference_model: StudentPerformanceModel = handler.load_object(
                        self.prediction_config.inference_model_s3_uri
                    )
            else:
                raise StudentPerformanceError(
                    "Neither local nor S3 inference model loading is enabled or configured properly.",
                    logger,
                )
        except Exception as e:
            logger.exception("Failed to load inference model.")
            raise StudentPerformanceError(e, logger) from e

    def predict(self, X_raw: pd.DataFrame) -> np.ndarray:
        try:
            logger.info("Running prediction using inference model.")
            y_pred = self.inference_model.predict(X_raw)
            return y_pred
        except Exception as e:
            logger.exception("Prediction failed.")
            raise StudentPerformanceError(e, logger) from e

    def predict_from_file(self, X_filepath: Path) -> np.ndarray:
        try:
            logger.info(f"Loading input data from file: {X_filepath}")
            X_raw = load_array(X_filepath, label="Input Data")
            return self.predict(X_raw)
        except Exception as e:
            logger.exception("Prediction from file failed.")
            raise StudentPerformanceError(e, logger) from e

    def save_predictions(self, predictions: np.ndarray) -> None:
        try:
            pred_df = pd.DataFrame({"prediction": predictions})
            file_name = f"{self.timestamp}_preds.csv"

            # Local save
            if self.prediction_config.local_enabled:
                local_dir = self.prediction_config.root_dir / self.timestamp
                local_dir.mkdir(parents=True, exist_ok=True)
                local_path = local_dir / file_name

                pred_df.to_csv(local_path, index=False)
                logger.info(f"Saved predictions locally at {local_path}")
            else:
                logger.info("Local save disabled — skipping local predictions save.")

            # S3 save
            if self.prediction_config.s3_enabled and self.backup_handler:
                s3_key = f"{self.prediction_config.root_s3_key}/{self.timestamp}/{file_name}"
                with self.backup_handler as handler:
                    handler.stream_df_as_csv(pred_df, s3_key)
                    logger.info(f"Saved predictions to S3 at {s3_key}")
            else:
                logger.info("S3 save disabled or backup handler not provided — skipping S3 save.")

        except Exception as e:
            logger.exception("Failed to save predictions.")
            raise StudentPerformanceError(e, logger) from e
