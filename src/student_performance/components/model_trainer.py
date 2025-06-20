from datetime import datetime, timezone
import os
import importlib
import numpy as np
import optuna
import mlflow
import dagshub
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer
from mlflow import sklearn as mlflow_sklearn
from pathlib import Path

from src.student_performance.entity.config_entity import ModelTrainerConfig
from src.student_performance.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger
from src.student_performance.utils.core import save_to_yaml, save_object, load_array
from src.student_performance.dbhandler.base_handler import DBHandler


class ModelTrainer:
    def __init__(
        self,
        config: ModelTrainerConfig,
        transformation_artifact: DataTransformationArtifact,
        backup_handler: DBHandler | None = None,
    ):
        try:
            logger.info(f"Initializing ModelTrainer with root_dir={config.root_dir}")
            self.config = config
            self.transformation_artifact = transformation_artifact
            self.backup_handler = backup_handler

            # ensure base directory exists
            config.root_dir.mkdir(parents=True, exist_ok=True)

            # set up MLflow if requested
            if config.tracking.mlflow.enabled:
                dagshub.init(
                    repo_owner=os.getenv("DAGSHUB_REPO_OWNER"),
                    repo_name=os.getenv("DAGSHUB_REPO_NAME"),
                    mlflow=True,
                )
                mlflow.set_tracking_uri(config.tracking.tracking_uri)
                mlflow.set_experiment(config.tracking.mlflow.experiment_name)

        except Exception as e:
            logger.exception("Failed to initialize ModelTrainer.")
            raise StudentPerformanceError(e, logger) from e

    def __load_data(self) -> None:
        try:
            logger.info("Loading train/val data")
            self.X_train = self.__load_array_flex(
                self.transformation_artifact.x_train_filepath,
                self.transformation_artifact.x_train_s3_uri,
                "X_train"
            )
            self.y_train = self.__load_array_flex(
                self.transformation_artifact.y_train_filepath,
                self.transformation_artifact.y_train_s3_uri,
                "y_train"
            )
            self.X_val = self.__load_array_flex(
                self.transformation_artifact.x_val_filepath,
                self.transformation_artifact.x_val_s3_uri,
                "X_val"
            )
            self.y_val = self.__load_array_flex(
                self.transformation_artifact.y_val_filepath,
                self.transformation_artifact.y_val_s3_uri,
                "y_val"
            )
        except Exception as e:
            logger.exception("Failed to load training data.")
            raise StudentPerformanceError(e, logger) from e

    def __load_array_flex(
        self,
        local_path: Path | None,
        s3_uri: str | None,
        label: str
    ) -> np.ndarray:
        try:
            if local_path and local_path.exists():
                logger.info(f"Loading {label} from local: {local_path}")
                return load_array(local_path, label)

            if s3_uri and self.backup_handler:
                logger.info(f"Loading {label} from S3: {s3_uri}")
                return self.backup_handler.load_array(s3_uri)

            raise ValueError(f"No valid source found for {label}")
        except Exception as e:
            logger.exception(f"Failed to load array for {label}")
            raise StudentPerformanceError(e, logger) from e

    def __select_and_tune(self) -> dict:
        try:
            logger.info("Selecting and (optionally) tuning models")
            best = {"score": -np.inf, "spec": None, "trial": None}

            for spec in self.config.models:
                if self.config.optimization.enabled:
                    trial, _ = self.__optimize_one(spec)
                    score = trial.value
                else:
                    model = self._instantiate(spec["name"], spec.get("params", {}))
                    score = cross_val_score(
                        model,
                        self.X_train,
                        self.y_train,
                        cv=self.config.optimization.cv_folds,
                        scoring=self.config.optimization.scoring
                    ).mean()
                if score > best["score"]:
                    best.update(score=score, spec=spec, trial=(trial if self.config.optimization.enabled else None))
            return best

        except Exception as e:
            logger.exception("Model selection/tuning failed.")
            raise StudentPerformanceError(e, logger) from e

    def __optimize_one(self, spec: dict):
        def objective(trial):
            params = {}
            for name, space in spec.get("search_space", {}).items():
                if "choices" in space:
                    params[name] = trial.suggest_categorical(name, space["choices"])
                else:
                    low, high = space["low"], space["high"]
                    if isinstance(low, int):
                        params[name] = trial.suggest_int(name, low, high, step=space.get("step", 1))
                    else:
                        params[name] = trial.suggest_float(name, low, high, log=space.get("log", False))
            clf = self._instantiate(spec["name"], params)
            return cross_val_score(
                clf, self.X_train, self.y_train,
                cv=self.config.optimization.cv_folds,
                scoring=self.config.optimization.scoring,
                n_jobs=-1,
            ).mean()

        study = optuna.create_study(direction=self.config.optimization.direction)
        study.optimize(objective, n_trials=self.config.optimization.n_trials)
        return study.best_trial, study

    def _instantiate(self, full_class_string: str, params: dict):
        module_path, cls_name = full_class_string.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)(**(params or {}))

    def __train_and_evaluate(self, spec: dict, params: dict):
        try:
            logger.info(f"Training final model: {spec['name']}")
            clf = self._instantiate(spec["name"], params)
            clf.fit(self.X_train, self.y_train)

            train_metrics = {
                m: get_scorer(m)(clf, self.X_train, self.y_train)
                for m in self.config.tracking.mlflow.metrics_to_log
            }
            val_metrics = {
                m: get_scorer(m)(clf, self.X_val, self.y_val)
                for m in self.config.tracking.mlflow.metrics_to_log
            }
            return clf, train_metrics, val_metrics

        except Exception as e:
            logger.exception("Final training/evaluation failed.")
            raise StudentPerformanceError(e, logger) from e

    def __save_artifacts(
        self,
        model,
        report: dict,
        inference_model,
    ) -> ModelTrainerArtifact:
        try:
            logger.info("Saving model and report artifacts locally")
            # trained model
            save_object(model, self.config.trained_model_filepath, "Trained Model")
            # inference wrapper
            save_object(inference_model, self.config.inference_model_filepath, "Inference Model")
            # report
            save_to_yaml(report, self.config.training_report_filepath, "Training Report")

            # optional S3 backup
            trained_s3 = report_s3 = inference_s3 = None
            if self.config.s3_enabled and self.backup_handler:
                logger.info("Streaming artifacts to S3")
                trained_s3 = self.backup_handler.stream_file(
                    self.config.trained_model_filepath.as_posix()
                )
                inference_s3 = self.backup_handler.stream_file(
                    self.config.inference_model_filepath.as_posix()
                )
                report_s3 = self.backup_handler.stream_file(
                    self.config.training_report_filepath.as_posix()
                )

            return ModelTrainerArtifact(
                trained_model_filepath=self.config.trained_model_filepath,
                training_report_filepath=self.config.training_report_filepath,
                trained_model_s3_uri=trained_s3,
                inference_model_s3_uri=inference_s3,
                report_s3_uri=report_s3,
                x_train_filepath=self.transformation_artifact.x_train_filepath,
                y_train_filepath=self.transformation_artifact.y_train_filepath,
                x_val_filepath=self.transformation_artifact.x_val_filepath,
                y_val_filepath=self.transformation_artifact.y_val_filepath,
                x_test_filepath=self.transformation_artifact.x_test_filepath,
                y_test_filepath=self.transformation_artifact.y_test_filepath,
            )
        except Exception as e:
            logger.exception("Failed to save artifacts.")
            raise StudentPerformanceError(e, logger) from e

    def run_training(self) -> ModelTrainerArtifact:
        try:
            logger.info("========== Starting Model Training ==========")
            self.__load_data()

            with mlflow.start_run():
                best = self.__select_and_tune()
                params = best["trial"].params if best["trial"] else best["spec"].get("params", {})
                model, train_m, val_m = self.__train_and_evaluate(best["spec"], params)

                # log to MLflow
                mlflow.log_params(params)
                for k, v in train_m.items():
                    mlflow.log_metric(f"train_{k}", v)
                for k, v in val_m.items():
                    mlflow.log_metric(f"val_{k}", v)

                # build inference wrapper
                inference_model = self.config.inference_model_class.from_objects(
                    model=model,
                    x_preprocessor=joblib.load(self.transformation_artifact.x_preprocessor_filepath),
                    y_preprocessor=joblib.load(self.transformation_artifact.y_preprocessor_filepath),
                )

                report = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "best_model": best["spec"]["name"].split(".")[-1],
                    "best_params": params,
                    "train_metrics": train_m,
                    "val_metrics": val_m,
                    "optimization": {
                        "enabled": self.config.optimization.enabled,
                        "best_score": best["score"],
                        "direction": self.config.optimization.direction,
                        "cv_folds": self.config.optimization.cv_folds,
                    },
                }

            logger.info("========== Model Training Completed ==========")
            return self.__save_artifacts(model, report, inference_model)

        except Exception as e:
            logger.exception("Model training pipeline failed.")
            raise StudentPerformanceError(e, logger) from e
