from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.student_performance.entity.config_entity import DataTransformationConfig
from src.student_performance.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)
from src.student_performance.logging import logger
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.utils.core import (
    read_csv,
    save_object,
    save_array,
)
from src.student_performance.data_processors.preprocessor_builder import PreprocessorBuilder
from src.student_performance.dbhandler.base_handler import DBHandler
from src.student_performance.constants.constants import (
    X_TRAIN_LABEL,
    Y_TRAIN_LABEL,
    X_VAL_LABEL,
    Y_VAL_LABEL,
    X_TEST_LABEL,
    Y_TEST_LABEL,
)


class DataTransformation:
    def __init__(
        self,
        transformation_config: DataTransformationConfig,
        validation_artifact: DataValidationArtifact,
        backup_handler: DBHandler | None = None,
    ):
        try:
            logger.info("Initializing DataTransformation component.")
            self.config = transformation_config
            self.validation_artifact = validation_artifact
            self.backup_handler = backup_handler

            self.df = self._load_validated_data(validation_artifact)

        except Exception as e:
            logger.exception("Failed to initialize DataTransformation.")
            raise StudentPerformanceError(e, logger) from e

    def _load_validated_data(
        self,
        validation_artifact: DataValidationArtifact
    ) -> pd.DataFrame:
        try:
            if self.config.local_enabled and validation_artifact.validated_filepath:
                logger.info(
                    f"Loading validated data from local: "
                    f"{validation_artifact.validated_filepath}",
                )
                return read_csv(validation_artifact.validated_filepath)

            if self.config.s3_enabled and validation_artifact.validated_s3_uri and self.backup_handler:
                logger.info(
                    f"Loading validated data from S3: "
                    f"{validation_artifact.validated_s3_uri}",
                )
                return self.backup_handler.load_csv(validation_artifact.validated_s3_uri)

            raise ValueError("No valid validated data source found.")
        except Exception as e:
            logger.exception("Failed to load validated data.")
            raise StudentPerformanceError(e, logger) from e

    def _split_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            X = self.df.drop(columns=self.config.target_column)
            y = self.df[self.config.target_column]
            return X, y
        except Exception as e:
            logger.exception("Failed to split features and target.")
            raise StudentPerformanceError(e, logger) from e

    def _split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        try:
            params = self.config.transformation_params.data_split
            stratify = y if params.stratify else None

            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                train_size=params.train_size,
                stratify=stratify,
                random_state=params.random_state,
            )

            test_ratio = params.test_size / (params.test_size + params.val_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=test_ratio,
                stratify=(y_temp if params.stratify else None),
                random_state=params.random_state,
            )

            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logger.exception("Failed to split data into train/val/test.")
            raise StudentPerformanceError(e, logger) from e

    def _save_preprocessors(self, x_proc, y_proc) -> None:
        try:
            # Save locally if enabled
            if self.config.local_enabled:
                save_object(
                    x_proc,
                    self.config.x_preprocessor_filepath,
                    label="X Preprocessor Pipeline",
                )
                save_object(
                    y_proc,
                    self.config.y_preprocessor_filepath,
                    label="Y Preprocessor Pipeline",
                )

            # Stream to S3 if enabled
            if self.config.s3_enabled and self.backup_handler:
                x_preprocessor_s3_key = self.config.x_preprocessor_s3_key
                y_preprocessor_s3_key = self.config.y_preprocessor_s3_key
                with self.backup_handler as handler:
                    handler.stream_object(
                        x_proc,
                        x_preprocessor_s3_key,
                    )

                    handler.stream_object(
                        y_proc,
                        y_preprocessor_s3_key,
                    )

        except Exception as e:
            logger.exception("Failed to save preprocessor pipelines.")
            raise StudentPerformanceError(e, logger) from e

    def _save_arrays(
        self,
        X_train, X_val, X_test,
        y_train, y_val, y_test
    ) -> None:
        try:
            # Local saves
            if self.config.local_enabled:
                to_save = [
                    (X_train, self.config.x_train_filepath, self.config.x_train_dvc_filepath,     X_TRAIN_LABEL),
                    (y_train, self.config.y_train_filepath, self.config.y_train_dvc_filepath,     Y_TRAIN_LABEL),
                    (X_val, self.config.x_val_filepath, self.config.x_val_dvc_filepath,       X_VAL_LABEL),
                    (y_val, self.config.y_val_filepath, self.config.y_val_dvc_filepath,       Y_VAL_LABEL),
                    (X_test, self.config.x_test_filepath, self.config.x_test_dvc_filepath,      X_TEST_LABEL),
                    (y_test, self.config.y_test_filepath, self.config.y_test_dvc_filepath,      Y_TEST_LABEL),
                ]
                for array, local_path, dvc_path, label in to_save:
                    save_array(array, local_path, dvc_path, label=label)

            # S3 backups
            if self.config.s3_enabled and self.backup_handler:
                to_stream = [
                        (X_train, self.config.x_train_s3_key),
                        (y_train, self.config.y_train_s3_key),
                        (X_val, self.config.x_val_s3_key),
                        (y_val, self.config.y_val_s3_key),
                        (X_test, self.config.x_test_s3_key),
                        (y_test, self.config.y_test_s3_key),
                    ]
                with self.backup_handler as handler:
                    for array, key in to_stream:
                        handler.stream_npy(array, key)

        except Exception as e:
            logger.exception("Failed to save transformed datasets.")
            raise StudentPerformanceError(e, logger) from e

    def run_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("========== Starting Data Transformation ==========")

            X, y = self._split_features_and_target()
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)

            builder = PreprocessorBuilder(
                steps=self.config.transformation_params.steps,
                methods=self.config.transformation_params.methods,
            )
            x_proc, y_proc = builder.build()

            X_train = x_proc.fit_transform(X_train)
            X_val   = x_proc.transform(X_val)
            X_test  = x_proc.transform(X_test)

            y_train = y_proc.fit_transform(y_train)
            y_val   = y_proc.transform(y_val)
            y_test  = y_proc.transform(y_test)

            self._save_preprocessors(x_proc, y_proc)
            self._save_arrays(X_train, X_val, X_test, y_train, y_val, y_test)

            logger.info("========== Data Transformation Completed ==========")
            return DataTransformationArtifact(
                x_train_filepath=self.config.x_train_filepath,
                y_train_filepath=self.config.y_train_filepath,
                x_val_filepath=self.config.x_val_filepath,
                y_val_filepath=self.config.y_val_filepath,
                x_test_filepath=self.config.x_test_filepath,
                y_test_filepath=self.config.y_test_filepath,
                x_preprocessor_filepath=self.config.x_preprocessor_filepath,
                y_preprocessor_filepath=self.config.y_preprocessor_filepath,
            )
        except Exception as e:
            logger.exception("Data transformation process failed.")
            raise StudentPerformanceError(e, logger) from e
