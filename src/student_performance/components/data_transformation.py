from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.student_performance.entity.config_entity import DataTransformationConfig
from src.student_performance.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.student_performance.logging import logger
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.utils.core import read_csv, save_object, save_array
from src.student_performance.data_processors.preprocessor_builder import PreprocessorBuilder
from src.student_performance.dbhandler.base_handler import DBHandler
from src.student_performance.constants.constants import (
    X_TRAIN_LABEL, Y_TRAIN_LABEL,
    X_VAL_LABEL, Y_VAL_LABEL,
    X_TEST_LABEL, Y_TEST_LABEL,
)
from box import ConfigBox


class DataTransformation:
    def __init__(
        self,
        transformation_config: DataTransformationConfig,
        validation_artifact: DataValidationArtifact,
        backup_handler: DBHandler | None = None,
    ):
        try:
            logger.info("Initializing DataTransformation component.")
            self.transformation_config = transformation_config
            self.validation_artifact = validation_artifact
            self.backup_handler = backup_handler

            self.df = self._load_validated_data(validation_artifact)
            logger.info("Validated dataset loaded successfully with shape: %s", self.df.shape)

        except Exception as e:
            logger.exception("Failed to initialize DataTransformation.")
            raise StudentPerformanceError(e, logger) from e

    def _load_validated_data(self, validation_artifact: DataValidationArtifact) -> pd.DataFrame:
        try:
            if self.transformation_config.local_enabled and validation_artifact.validated_filepath:
                logger.info("Loading validated data from local path: %s", validation_artifact.validated_filepath)
                return read_csv(validation_artifact.validated_filepath)

            if self.transformation_config.s3_enabled and validation_artifact.validated_s3_uri and self.backup_handler:
                logger.info("Loading validated data from S3 URI: %s", validation_artifact.validated_s3_uri)
                return self.backup_handler.load_csv(validation_artifact.validated_s3_uri)

            raise ValueError("No valid validated data source found.")
        except Exception as e:
            logger.exception("Failed to load validated data.")
            raise StudentPerformanceError(e, logger) from e

    def _split_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            logger.info("Splitting features and target column: '%s'", self.transformation_config.target_column)
            X = self.df.drop(columns=self.transformation_config.target_column)
            y = self.df[self.transformation_config.target_column]
            logger.info("Feature shape: %s | Target shape: %s", X.shape, y.shape)
            return X, y
        except Exception as e:
            logger.exception("Failed to split features and target.")
            raise StudentPerformanceError(e, logger) from e

    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        try:
            params = self.transformation_config.transformation_params.data_split
            stratify = y if params.stratify else None

            logger.info("Splitting into training and temp sets with train_size=%.2f", params.train_size)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                train_size=params.train_size,
                stratify=stratify,
                random_state=params.random_state,
            )

            test_ratio = params.test_size / (params.test_size + params.val_size)
            logger.info("Splitting temp set into validation and test with test_ratio=%.2f", test_ratio)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=test_ratio,
                stratify=(y_temp if params.stratify else None),
                random_state=params.random_state,
            )

            logger.info("Final splits — Train: %s | Val: %s | Test: %s", X_train.shape, X_val.shape, X_test.shape)
            return X_train, X_val, X_test, y_train, y_val, y_test
        except Exception as e:
            logger.exception("Failed to split data into train/val/test.")
            raise StudentPerformanceError(e, logger) from e

    def _save_preprocessors(self, x_proc, y_proc) -> ConfigBox:
        result = ConfigBox({
            "x_preprocessor": {"local": None, "s3": None},
            "y_preprocessor": {"local": None, "s3": None},
        })
        try:
            if self.transformation_config.local_enabled:
                logger.info("Saving X and Y preprocessors locally.")
                save_object(x_proc, self.transformation_config.x_preprocessor_filepath, label="X Preprocessor Pipeline")
                result.x_preprocessor.local = self.transformation_config.x_preprocessor_filepath

                save_object(y_proc, self.transformation_config.y_preprocessor_filepath, label="Y Preprocessor Pipeline")
                result.y_preprocessor.local = self.transformation_config.y_preprocessor_filepath

            if self.transformation_config.s3_enabled and self.backup_handler:
                logger.info("Streaming X and Y preprocessors to S3.")
                with self.backup_handler as handler:
                    result.x_preprocessor.s3 = handler.stream_object(x_proc, self.transformation_config.x_preprocessor_s3_key)
                    result.y_preprocessor.s3 = handler.stream_object(y_proc, self.transformation_config.y_preprocessor_s3_key)

            return result
        except Exception as e:
            logger.exception("Failed to save preprocessor pipelines.")
            raise StudentPerformanceError(e, logger) from e

    def _save_arrays(self, X_train, X_val, X_test, y_train, y_val, y_test) -> ConfigBox:
        result = ConfigBox({
            "X_train": {"local": None, "s3": None},
            "y_train": {"local": None, "s3": None},
            "X_val":   {"local": None, "s3": None},
            "y_val":   {"local": None, "s3": None},
            "X_test":  {"local": None, "s3": None},
            "y_test":  {"local": None, "s3": None},
        })
        try:
            if self.transformation_config.local_enabled:
                logger.info("Saving transformed arrays locally.")
                to_save = [
                    ("X_train", X_train, self.transformation_config.x_train_filepath, self.transformation_config.x_train_dvc_filepath, X_TRAIN_LABEL),
                    ("y_train", y_train, self.transformation_config.y_train_filepath, self.transformation_config.y_train_dvc_filepath, Y_TRAIN_LABEL),
                    ("X_val", X_val, self.transformation_config.x_val_filepath, self.transformation_config.x_val_dvc_filepath, X_VAL_LABEL),
                    ("y_val", y_val, self.transformation_config.y_val_filepath, self.transformation_config.y_val_dvc_filepath, Y_VAL_LABEL),
                    ("X_test", X_test, self.transformation_config.x_test_filepath, self.transformation_config.x_test_dvc_filepath, X_TEST_LABEL),
                    ("y_test", y_test, self.transformation_config.y_test_filepath, self.transformation_config.y_test_dvc_filepath, Y_TEST_LABEL),
                ]
                for key, array, local_path, dvc_path, label in to_save:
                    logger.info("Saving %s | Shape: %s", label, array.shape)
                    save_array(array, local_path, dvc_path, label=label)
                    result[key]["local"] = local_path

            if self.transformation_config.s3_enabled and self.backup_handler:
                logger.info("Streaming arrays to S3.")
                to_stream = [
                    ("X_train", X_train, self.transformation_config.x_train_s3_key),
                    ("y_train", y_train, self.transformation_config.y_train_s3_key),
                    ("X_val", X_val, self.transformation_config.x_val_s3_key),
                    ("y_val", y_val, self.transformation_config.y_val_s3_key),
                    ("X_test", X_test, self.transformation_config.x_test_s3_key),
                    ("y_test", y_test, self.transformation_config.y_test_s3_key),
                    ("X_train", X_train, self.transformation_config.x_train_dvc_s3_key),
                    ("y_train", y_train, self.transformation_config.y_train_dvc_s3_key),
                    ("X_val", X_val, self.transformation_config.x_val_dvc_s3_key),
                    ("y_val", y_val, self.transformation_config.y_val_dvc_s3_key),
                    ("X_test", X_test, self.transformation_config.x_test_dvc_s3_key),
                    ("y_test", y_test, self.transformation_config.y_test_dvc_s3_key),
                ]
                with self.backup_handler as handler:
                    for key, array, s3_key in to_stream:
                        uri = handler.stream_npy(array, s3_key)
                        result[key]["s3"] = uri

            return result
        except Exception as e:
            logger.exception("Failed to save transformed datasets.")
            raise StudentPerformanceError(e, logger) from e

    def run_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("========== Starting Data Transformation ==========")

            X, y = self._split_features_and_target()
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)

            builder = PreprocessorBuilder(
                steps=self.transformation_config.transformation_params.steps,
                methods=self.transformation_config.transformation_params.methods,
            )
            x_proc, y_proc = builder.build()

            logger.info("Fitting X and Y preprocessing pipelines.")
            X_train = x_proc.fit_transform(X_train)
            X_val = x_proc.transform(X_val)
            X_test = x_proc.transform(X_test)

            y_train = y_proc.fit_transform(y_train)
            y_val = y_proc.transform(y_val)
            y_test = y_proc.transform(y_test)

            # Convert sparse matrices if necessary
            X_train = X_train.toarray() if hasattr(X_train, "toarray") else X_train
            X_val = X_val.toarray() if hasattr(X_val, "toarray") else X_val
            X_test = X_test.toarray() if hasattr(X_test, "toarray") else X_test

            prep_locs = self._save_preprocessors(x_proc, y_proc)
            array_locs = self._save_arrays(X_train, X_val, X_test, y_train, y_val, y_test)

            logger.info("========== Data Transformation Completed ==========")

            return DataTransformationArtifact(
                x_train_filepath=array_locs.X_train.local,
                y_train_filepath=array_locs.y_train.local,
                x_val_filepath=array_locs.X_val.local,
                y_val_filepath=array_locs.y_val.local,
                x_test_filepath=array_locs.X_test.local,
                y_test_filepath=array_locs.y_test.local,
                x_preprocessor_filepath=prep_locs.x_preprocessor.local,
                y_preprocessor_filepath=prep_locs.y_preprocessor.local,
                x_train_s3_uri=array_locs.X_train.s3,
                y_train_s3_uri=array_locs.y_train.s3,
                x_val_s3_uri=array_locs.X_val.s3,
                y_val_s3_uri=array_locs.y_val.s3,
                x_test_s3_uri=array_locs.X_test.s3,
                y_test_s3_uri=array_locs.y_test.s3,
                x_preprocessor_s3_uri=prep_locs.x_preprocessor.s3,
                y_preprocessor_s3_uri=prep_locs.y_preprocessor.s3,
            )
        except Exception as e:
            logger.exception("Data transformation process failed.")
            raise StudentPerformanceError(e, logger) from e
