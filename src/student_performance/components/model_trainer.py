from datetime import datetime, timezone
import os
import importlib
import numpy as np
import optuna
import mlflow
import dagshub
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer, r2_score
from mlflow import sklearn as mlflow_sklearn
from pathlib import Path

from src.student_performance.entity.config_entity import ModelTrainerConfig
from src.student_performance.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
)
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger
from src.student_performance.utils.core import (
    save_to_yaml,
    save_object,
    load_array,
    load_object,
)
from src.student_performance.dbhandler.base_handler import DBHandler
from src.student_performance.inference.estimator import StudentPerformanceModel


class ModelTrainer:
    def __init__(
        self,
        trainer_config: ModelTrainerConfig,
        transformation_artifact: DataTransformationArtifact,
        backup_handler: DBHandler | None = None,
    ):
        try:
            self.trainer_config = trainer_config
            self.transformation_artifact = transformation_artifact
            self.backup_handler = backup_handler

            if trainer_config.tracking.mlflow.enabled:
                dagshub.init(
                    repo_owner=os.getenv("DAGSHUB_REPO_OWNER"),
                    repo_name=os.getenv("DAGSHUB_REPO_NAME"),
                    mlflow=True,
                )
                mlflow.set_tracking_uri(trainer_config.tracking.tracking_uri)
                mlflow.set_experiment(
                    trainer_config.tracking.mlflow.experiment_name,
                )
        except Exception as e:
            logger.exception("Failed to initialize ModelTrainer.")
            raise StudentPerformanceError(e, logger) from e

    def __load_data(self) -> None:
        try:
            logger.info("Loading train/val data")

            if self.trainer_config.local_enabled:
                logger.info("Loading from local")
                self.X_train = load_array(self.transformation_artifact.x_train_filepath, "X_train")
                self.y_train = load_array(self.transformation_artifact.y_train_filepath, "y_train")
                self.X_val = load_array(self.transformation_artifact.x_val_filepath, "X_val")
                self.y_val = load_array(self.transformation_artifact.y_val_filepath, "y_val")
                self.x_preprocessor = load_object(self.transformation_artifact.x_preprocessor_filepath, "X_Processor")
                self.y_preprocessor = load_object(self.transformation_artifact.y_preprocessor_filepath, "Y_Processor")

            elif self.trainer_config.s3_enabled and self.backup_handler:
                logger.info("Loading from S3")
                with self.backup_handler as handler:
                    self.X_train = handler.load_npy(self.transformation_artifact.x_train_s3_uri)
                    self.y_train = handler.load_npy(self.transformation_artifact.y_train_s3_uri)
                    self.X_val = handler.load_npy(self.transformation_artifact.x_val_s3_uri)
                    self.y_val = handler.load_npy(self.transformation_artifact.y_val_s3_uri)
                    self.x_preprocessor = handler.load_object(self.transformation_artifact.x_preprocessor_s3_uri)
                    self.y_preprocessor = handler.load_object(self.transformation_artifact.y_preprocessor_s3_uri)

        except Exception as e:
            logger.exception("Failed to load training data.")
            raise StudentPerformanceError(e, logger) from e

    @staticmethod
    def compute_adjusted_r2(r2: float, n_samples: int, n_features: int) -> float:
        if n_samples <= n_features + 1:
            return np.nan
        return 1 - (1 - r2) * ((n_samples - 1) / (n_samples - n_features - 1))

    def __select_and_tune(self) -> dict:
        try:
            logger.info("Selecting and (optionally) tuning models")
            best = {"score": -np.inf, "spec": None, "trial": None}

            for spec in self.trainer_config.models:
                if self.trainer_config.optimization.enabled:
                    trial, _ = self.__optimize_one(spec)
                    score = trial.value
                else:
                    model = self._instantiate(spec["name"], spec.get("params", {}))
                    score = cross_val_score(
                        model,
                        self.X_train,
                        self.y_train,
                        cv=self.trainer_config.optimization.cv_folds,
                        scoring=self.trainer_config.optimization.scoring,
                    ).mean()
                if score > best["score"]:
                    best.update(
                        score=score,
                        spec=spec,
                        trial=trial if self.trainer_config.optimization.enabled else None,
                    )
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
                clf,
                self.X_train,
                self.y_train,
                cv=self.trainer_config.optimization.cv_folds,
                scoring=self.trainer_config.optimization.scoring,
                n_jobs=-1,
            ).mean()

        study = optuna.create_study(direction=self.trainer_config.optimization.direction)
        study.optimize(objective, n_trials=self.trainer_config.optimization.n_trials)
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

            def compute_metrics(X, y):
                metrics = {}
                for m in self.trainer_config.tracking.mlflow.metrics_to_log:
                    if m == "adjusted_r2":
                        r2 = r2_score(y, clf.predict(X))
                        adj_r2 = self.compute_adjusted_r2(r2, X.shape[0], X.shape[1])
                        metrics["adjusted_r2"] = adj_r2
                    else:
                        scorer = get_scorer(m)
                        metrics[m] = scorer(clf, X, y)
                return metrics

            train_metrics = compute_metrics(self.X_train, self.y_train)
            val_metrics = compute_metrics(self.X_val, self.y_val)

            return clf, train_metrics, val_metrics

        except Exception as e:
            logger.exception("Final training/evaluation failed.")
            raise StudentPerformanceError(e, logger) from e

    def __save_artifacts(
        self,
        model,
        report: dict,
        inference_model,
        experiment_id: str,
        run_id: str,
    ) -> ModelTrainerArtifact:
        trained_local = None
        report_local = None
        inference_local = None

        trained_s3 = None
        report_s3 = None
        inference_s3 = None

        if self.trainer_config.local_enabled:
            trained_local = self.trainer_config.trained_model_filepath
            save_object(model, trained_local, label="Trained Model")

            inference_local = self.trainer_config.inference_model_filepath
            save_object(inference_model, inference_local, label="Inference Model")

            report_local = self.trainer_config.training_report_filepath
            save_to_yaml(report, report_local, label="Training Report")

        if self.trainer_config.s3_enabled and self.backup_handler:
            logger.info("Streaming artifacts to S3")
            with self.backup_handler as handler:
                trained_s3 = handler.stream_object(model, self.trainer_config.trained_model_s3_key)
                inference_s3 = handler.stream_object(inference_model, self.trainer_config.inference_model_s3_key)
                report_s3 = handler.stream_yaml(report, self.trainer_config.training_report_s3_key)

        return ModelTrainerArtifact(
            trained_model_filepath=trained_local,
            training_report_filepath=report_local,
            inference_model_filepath=inference_local,
            x_train_filepath=self.transformation_artifact.x_train_filepath if self.trainer_config.local_enabled else None,
            y_train_filepath=self.transformation_artifact.y_train_filepath if self.trainer_config.local_enabled else None,
            x_val_filepath=self.transformation_artifact.x_val_filepath if self.trainer_config.local_enabled else None,
            y_val_filepath=self.transformation_artifact.y_val_filepath if self.trainer_config.local_enabled else None,
            x_test_filepath=self.transformation_artifact.x_test_filepath if self.trainer_config.local_enabled else None,
            y_test_filepath=self.transformation_artifact.y_test_filepath if self.trainer_config.local_enabled else None,
            trained_model_s3_uri=trained_s3,
            training_report_s3_uri=report_s3,
            inference_model_s3_uri=inference_s3,
            x_train_s3_uri=self.transformation_artifact.x_train_s3_uri if self.trainer_config.s3_enabled else None,
            y_train_s3_uri=self.transformation_artifact.y_train_s3_uri if self.trainer_config.s3_enabled else None,
            x_val_s3_uri=self.transformation_artifact.x_val_s3_uri if self.trainer_config.s3_enabled else None,
            y_val_s3_uri=self.transformation_artifact.y_val_s3_uri if self.trainer_config.s3_enabled else None,
            x_test_s3_uri=self.transformation_artifact.x_test_s3_uri if self.trainer_config.s3_enabled else None,
            y_test_s3_uri=self.transformation_artifact.y_test_s3_uri if self.trainer_config.s3_enabled else None,
            experiment_id=experiment_id,
            run_id=run_id,
        )

    def __to_positive(self, metrics: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, val in metrics.items():
            if name.startswith("neg_"):
                out[name[4:]] = -val
            else:
                out[name] = val
        return out

    def run_training(self) -> ModelTrainerArtifact:
        try:
            logger.info("========== Starting Model Training ==========")
            self.__load_data()

            with mlflow.start_run() as active_run:
                exp_id = active_run.info.experiment_id
                run_id = active_run.info.run_id

                best = self.__select_and_tune()
                params = best["trial"].params if best["trial"] else best["spec"].get("params", {})

                model, train_m, val_m = self.__train_and_evaluate(best["spec"], params)

                train_pos = self.__to_positive(train_m)
                val_pos = self.__to_positive(val_m)

                mlflow.log_params(params)
                for name, val in train_pos.items():
                    mlflow.log_metric(f"train_{name}", val)
                for name, val in val_pos.items():
                    mlflow.log_metric(f"val_{name}", val)

                inference_model = StudentPerformanceModel.from_objects(
                    model=model,
                    x_preprocessor=self.x_preprocessor,
                    y_preprocessor=self.y_preprocessor,
                )

                report = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "best_model": best["spec"]["name"].split(".")[-1],
                    "best_params": params,
                    "train_metrics": train_pos,
                    "val_metrics": val_pos,
                    "optimization": {
                        "enabled": self.trainer_config.optimization.enabled,
                        "best_score": best["score"],
                        "direction": self.trainer_config.optimization.direction,
                        "cv_folds": self.trainer_config.optimization.cv_folds,
                    },
                }

            logger.info("========== Model Training Completed ==========")
            return self.__save_artifacts(model, report, inference_model, exp_id, run_id)

        except Exception as e:
            logger.exception("Model training pipeline failed.")
            raise StudentPerformanceError(e, logger) from e
