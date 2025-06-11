# FILE: src/student_performance/components/data_transformation.py

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
from src.student_performance.constants.constants import (
    X_TRAIN_LABEL, Y_TRAIN_LABEL,
    X_VAL_LABEL, Y_VAL_LABEL,
    X_TEST_LABEL, Y_TEST_LABEL
)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, validation_artifact: DataValidationArtifact):
        try:
            self.config = config
            self.validation_artifact = validation_artifact
            self.df = read_csv(validation_artifact.validated_filepath)
        except Exception as e:
            raise StudentPerformanceError(e, logger) from e

    def _split_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            df = self.df.copy()
            X = df.drop(columns=[self.config.target_column])
            y = df[self.config.target_column]
            return X, y
        except Exception as e:
            raise StudentPerformanceError(e, logger) from e

    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        try:
            split_params = self.config.transformation_params.data_split
            stratify = y if split_params.stratify else None

            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                train_size=split_params.train_size,
                stratify=stratify,
                random_state=split_params.random_state
            )

            relative_test_size = split_params.test_size / (split_params.test_size + split_params.val_size)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=relative_test_size,
                stratify=y_temp if split_params.stratify else None,
                random_state=split_params.random_state
            )

            return X_train, X_val, X_test, y_train, y_val, y_test
        except Exception as e:
            raise StudentPerformanceError(e, logger) from e

    def _save_datasets(self, X_train, X_val, X_test, y_train, y_val, y_test):
        try:
            save_array(X_train, self.config.x_train_filepath, self.config.x_train_dvc_filepath, label=X_TRAIN_LABEL)
            save_array(y_train, self.config.y_train_filepath, self.config.y_train_dvc_filepath, label=Y_TRAIN_LABEL)
            save_array(X_val, self.config.x_val_filepath, self.config.x_val_dvc_filepath, label=X_VAL_LABEL)
            save_array(y_val, self.config.y_val_filepath, self.config.y_val_dvc_filepath, label=Y_VAL_LABEL)
            save_array(X_test, self.config.x_test_filepath, self.config.x_test_dvc_filepath, label=X_TEST_LABEL)
            save_array(y_test, self.config.y_test_filepath, self.config.y_test_dvc_filepath, label=Y_TEST_LABEL)
        except Exception as e:
            raise StudentPerformanceError(e, logger) from e

    def run_transformation(self) -> DataTransformationArtifact:
        try:
            logger.info("========== Starting Data Transformation ==========")

            # Step 1: Separate features and target
            X, y = self._split_features_and_target()

            # Step 2: Split data into train/val/test
            X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)

            # Step 3: Build preprocessor pipelines for X and Y
            builder = PreprocessorBuilder(
                steps=self.config.transformation_params.steps,
                methods=self.config.transformation_params.methods,
            )
            x_processor, y_processor = builder.build()

            # Step 4: Fit and transform data
            X_train = x_processor.fit_transform(X_train)
            X_val = x_processor.transform(X_val)
            X_test = x_processor.transform(X_test)

            y_train = y_processor.fit_transform(y_train)
            y_val = y_processor.transform(y_val)
            y_test = y_processor.transform(y_test)

            # Step 5: Save X and Y processors
            save_object(x_processor, self.config.x_preprocessor_filepath, label="X Preprocessor Pipeline")
            save_object(y_processor, self.config.y_preprocessor_filepath, label="Y Preprocessor Pipeline")

            # Step 6: Save transformed datasets
            self._save_datasets(X_train, X_val, X_test, y_train, y_val, y_test)

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
            logger.error("Data transformation failed.")
            raise StudentPerformanceError(e, logger) from e
