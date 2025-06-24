from typing import Any, Dict
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    explained_variance_score,
    r2_score,
)
import mlflow
from pathlib import Path
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger
from src.student_performance.utils.core import save_to_yaml
from src.student_performance.dbhandler.base_handler import DBHandler
from src.student_performance.entity.config_entity import ModelEvaluationConfig
from src.student_performance.entity.artifact_entity import (
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
)
from src.student_performance.utils.core import (
    save_to_yaml,
    load_array,
    load_object,
)

class ModelEvaluation:
    def __init__(
        self,
        evaluation_config: ModelEvaluationConfig,
        trainer_artifact: ModelTrainerArtifact,
        backup_handler: DBHandler | None = None,
    ):
        try:
            logger.info("Initializing ModelEvaluation.")
            self.evaluation_config = evaluation_config
            self.trainer_artifact = trainer_artifact
            self.backup_handler = backup_handler

            # Load model and test data
            self._load_data()

            run_id = self.trainer_artifact.run_id

            # Set up MLflow tracking if enabled
            if self.evaluation_config.tracking.mlflow.enabled:
                logger.info(f"Resuming MLflow run {run_id}")
                mlflow.start_run(run_id=run_id)

        except Exception as e:
            logger.exception("Failed to initialize ModelEvaluation.")
            raise StudentPerformanceError(e, logger) from e

    def _load_data(self):
        """Load model, X_test, and y_test based on local_enabled / s3_enabled flags."""
        try:
            if self.evaluation_config.local_enabled:
                logger.info("Loading model and data from local paths.")

                if not self.trainer_artifact.trained_model_filepath:
                    raise StudentPerformanceError("Trained model local path missing.", logger)
                self.model = load_object(self.trainer_artifact.trained_model_filepath, label="Trained Model")

                if not self.trainer_artifact.x_test_filepath:
                    raise StudentPerformanceError("X_test local path missing.", logger)
                self.x_test = load_array(self.trainer_artifact.x_test_filepath, label="X_test")

                if not self.trainer_artifact.y_test_filepath:
                    raise StudentPerformanceError("y_test local path missing.", logger)
                self.y_test = load_array(self.trainer_artifact.y_test_filepath, label="y_test")

            elif self.evaluation_config.s3_enabled and self.backup_handler:
                logger.info("Loading model and data from S3 URIs.")
                with self.backup_handler as handler:
                    if not self.trainer_artifact.trained_model_s3_uri:
                        raise StudentPerformanceError("Trained model S3 URI missing.", logger)
                    self.model = handler.load_object(self.trainer_artifact.trained_model_s3_uri)

                    if not self.trainer_artifact.x_test_s3_uri:
                        raise StudentPerformanceError("X_test S3 URI missing.", logger)
                    self.x_test = handler.load_npy(self.trainer_artifact.x_test_s3_uri)

                    if not self.trainer_artifact.y_test_s3_uri:
                        raise StudentPerformanceError("y_test S3 URI missing.", logger)
                    self.y_test = handler.load_npy(self.trainer_artifact.y_test_s3_uri)

            else:
                raise StudentPerformanceError("Neither local nor S3 loading is enabled or configured properly.", logger)

        except Exception as e:
            logger.exception("Failed during data loading in ModelEvaluation.")
            raise StudentPerformanceError(e, logger) from e

    @staticmethod
    def compute_adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
        if n_samples <= n_features + 1:
            return np.nan
        return 1 - (1 - r2) * ((n_samples - 1) / (n_samples - n_features - 1))

    def _save_report(self, results: Dict[str, float]) -> None:
        """Save the evaluation report locally or to S3 based on config."""
        try:
            if self.evaluation_config.local_enabled:
                report_path = self.evaluation_config.evaluation_report_filepath
                save_to_yaml(results, str(report_path), label="Evaluation Report")
                logger.info(f"Saved evaluation report locally at {report_path}")

            if self.evaluation_config.s3_enabled and self.backup_handler:
                with self.backup_handler as handler:
                    report_s3_uri = handler.stream_yaml(
                        results,
                        self.evaluation_config.evaluation_report_s3_key
                    )
                    logger.info(f"Saved evaluation report to S3 at {report_s3_uri}")

        except Exception as e:
            logger.exception("Failed to save evaluation report.")
            raise StudentPerformanceError(e, logger) from e

    def run_evaluation(self) -> Dict[str, Any]:
        try:
            logger.info("Starting model evaluation.")
            y_pred = self.model.predict(self.x_test)
            n_samples, n_features = self.x_test.shape

            eval_metrics = self.evaluation_config.eval_metrics.metrics
            results: Dict[str, float] = {}

            for metric in eval_metrics:
                if metric == "mean_absolute_error":
                    results["mean_absolute_error"] = mean_absolute_error(self.y_test, y_pred)
                elif metric == "mean_squared_error":
                    results["mean_squared_error"] = mean_squared_error(self.y_test, y_pred)
                elif metric == "root_mean_squared_error":
                    results["root_mean_squared_error"] = np.sqrt(mean_squared_error(self.y_test, y_pred))
                elif metric == "median_absolute_error":
                    results["median_absolute_error"] = median_absolute_error(self.y_test, y_pred)
                elif metric == "explained_variance_score":
                    results["explained_variance_score"] = explained_variance_score(self.y_test, y_pred)
                elif metric == "r2":
                    results["r2"] = r2_score(self.y_test, y_pred)
                elif metric == "adjusted_r2":
                    r2_val = r2_score(self.y_test, y_pred)
                    results["adjusted_r2"] = self.compute_adjusted_r2(r2_val, n_samples, n_features)
                else:
                    logger.warning(f"Unsupported metric: {metric}")

            logger.info(f"Evaluation results: {results}")

            if self.evaluation_config.tracking.mlflow.enabled:
                for k, v in results.items():
                    mlflow.log_metric(f"test_{k}", v)

            self._save_report(results)

            return results

        except Exception as e:
            logger.exception("Model evaluation failed.")
            raise StudentPerformanceError(e, logger) from e
