import hashlib
import pandas as pd
from box import ConfigBox
from scipy.stats import ks_2samp
import sys

from src.student_performance.entity.config_entity import DataValidationConfig
from src.student_performance.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger
from src.student_performance.utils.core import (
    read_csv, save_to_csv, save_to_yaml
)
from src.student_performance.utils.timestamp import get_utc_timestamp


class DataValidation:
    """
    Validates input data using schema checks, missing value checks,
    duplicate removal, categorical value enforcement, and drift detection.
    Generates validation reports and outputs a DataValidationArtifact.
    """

    def __init__(self, validation_config: DataValidationConfig, ingestion_artifact: DataIngestionArtifact):
        """
        Initialize data validator with configuration and ingested data.
        Loads base data if drift detection is enabled.
        """
        try:
            logger.info("Initializing DataValidation component.")

            # Load validation schema and parameters
            self.validation_config = validation_config
            self.schema = self.validation_config.schema
            self.params = self.validation_config.validation_params

            # Load ingested data
            self.df = read_csv(ingestion_artifact.ingested_data_filepath)
            self.base_df = None
            self.drift_check_performed = False
            self.timestamp = get_utc_timestamp()

            # Load base data for drift comparison, if enabled
            if self.params.drift_detection.enabled and self.validation_config.dvc_validated_filepath.exists():
                self.base_df = read_csv(self.validation_config.dvc_validated_filepath)
                self.drift_check_performed = True

            # Prepare report templates
            self.report = ConfigBox(self.validation_config.report_template.copy())
            self.critical_checks = ConfigBox({k: False for k in self.report.check_results.critical_checks.keys()})
            self.non_critical_checks = ConfigBox({k: False for k in self.report.check_results.non_critical_checks.keys()})

        except Exception as e:
            raise StudentPerformanceError(e, logger) from e

    def __check_schema_hash(self):
        """
        Compare the hash of the expected schema with the actual schema from the dataset.
        """
        try:
            logger.info("Performing schema hash check.")

            # Compute expected hash
            expected = self.schema.columns
            expected_str = "|".join(f"{col}:{dtype}" for col, dtype in sorted(expected.items()))
            expected_hash = hashlib.md5(expected_str.encode()).hexdigest()

            # Compute current dataset hash
            current_str = "|".join(f"{col}:{self.df[col].dtype}" for col in sorted(self.df.columns))
            current_hash = hashlib.md5(current_str.encode()).hexdigest()

            self.critical_checks.schema_is_match = (expected_hash == current_hash)

            logger.info("Schema hash check passed." if self.critical_checks.schema_is_match else "Schema hash mismatch.")
        except Exception as e:
            logger.exception("Schema hash check failed.")
            self.critical_checks.schema_is_match = False
            raise StudentPerformanceError(e, logger) from e

    def __check_schema_structure(self):
        """
        Check if dataset columns structurally match expected schema.
        """
        try:
            logger.info("Performing schema structure check.")

            expected_cols = set(self.schema.columns.keys()) | {self.schema.target_column}
            actual_cols = set(self.df.columns)

            self.critical_checks.schema_is_match = (expected_cols == actual_cols)

            if not self.critical_checks.schema_is_match:
                logger.error(f"Schema structure mismatch: expected={expected_cols}, actual={actual_cols}")
        except Exception as e:
            logger.exception("Schema structure check failed.")
            self.critical_checks.schema_is_match = False
            raise StudentPerformanceError(e, logger) from e

    def __check_missing_values(self):
        """
        Checks for missing values in the dataset and logs a YAML report.
        """
        try:
            logger.info("Checking for missing values.")

            missing = self.df.isnull().sum().to_dict()
            missing["timestamp"] = self.timestamp

            save_to_yaml(missing, self.validation_config.missing_report_filepath, label="Missing Value Report")

            self.non_critical_checks.no_missing_values = not any(
                v > 0 for v in missing.values() if isinstance(v, (int, float))
            )

            logger.info("Missing value check passed." if self.non_critical_checks.no_missing_values else "Missing values detected.")
        except Exception as e:
            logger.exception("Missing value check failed.")
            self.non_critical_checks.no_missing_values = False
            raise StudentPerformanceError(e, logger) from e

    def __check_duplicates(self):
        """
        Removes duplicates and logs the number removed.
        """
        try:
            logger.info("Checking for duplicate rows.")

            before = len(self.df)
            self.df = self.df.drop_duplicates()
            after = len(self.df)

            duplicates_removed = before - after

            result = {
                "duplicate_rows_removed": duplicates_removed,
                "timestamp": self.timestamp
            }

            save_to_yaml(result, self.validation_config.duplicates_report_filepath, label="Duplicates Report")
            self.non_critical_checks.no_duplicate_rows = (duplicates_removed == 0)

            logger.info("Duplicate check passed." if duplicates_removed == 0 else f"{duplicates_removed} duplicates removed.")
        except Exception as e:
            logger.exception("Duplicate check failed.")
            self.non_critical_checks.no_duplicate_rows = False
            raise StudentPerformanceError(e, logger) from e

    def __check_categorical_values(self):
        """
        Checks whether categorical columns contain only allowed values.
        """
        try:
            logger.info("Checking categorical values against schema constraints.")

            allowed_values = self.schema.get("allowed_values", {})
            violations = {}

            for col, allowed in allowed_values.items():
                if col not in self.df.columns:
                    continue

                actual_values = set(self.df[col].dropna().unique())
                unexpected = actual_values - set(allowed)

                if unexpected:
                    violations[col] = {
                        "unexpected_values": sorted(list(unexpected)),
                        "expected_values": allowed
                    }

            result = {
                "violations_found": bool(violations),
                "details": violations,
                "timestamp": self.timestamp
            }

            save_to_yaml(result, self.validation_config.categorical_report_filepath, label="Categorical Values Report")
            self.non_critical_checks.categorical_values_match = not bool(violations)

            logger.info("Categorical values check passed." if not violations else "Categorical values mismatch found.")
        except Exception as e:
            logger.exception("Categorical value check failed.")
            self.non_critical_checks.categorical_values_match = False
            raise StudentPerformanceError(e, logger) from e

    def __check_data_drift(self):
        """
        Performs KS test between base and current datasets to detect drift.
        """
        try:
            logger.info("Performing data drift check.")

            drift_results = {
                "timestamp": self.timestamp,
                "drift_check_performed": False
            }

            if self.base_df is None:
                logger.info("Base data not available. Skipping drift check.")
                drift_results["reason"] = "Base dataset not found. Drift check skipped."
                save_to_yaml(drift_results, self.validation_config.drift_report_filepath, label="Drift Report")
                self.critical_checks.no_data_drift = True
                return

            drift_detected = False
            drift_results.update({
                "drift_check_performed": True,
                "drift_method": self.params.drift_detection.method,
                "columns": {}
            })

            for col in self.df.columns:
                if col not in self.base_df.columns:
                    continue

                _, p_value = ks_2samp(self.base_df[col], self.df[col])
                drift = p_value < self.params.drift_detection.p_value_threshold
                drift_results["columns"][col] = {"p_value": float(p_value), "drift": drift}

                if drift:
                    drift_detected = True

            drift_results["drift_detected"] = drift_detected
            save_to_yaml(drift_results, self.validation_config.drift_report_filepath, label="Drift Report")
            self.critical_checks.no_data_drift = not drift_detected

            logger.info("Drift check passed." if not drift_detected else "Drift detected in columns.")
        except Exception as e:
            logger.exception("Data drift check failed.")
            self.critical_checks.no_data_drift = False
            raise StudentPerformanceError(e, logger) from e

    def __generate_report(self) -> dict:
        """
        Generate a unified validation report from individual checks.
        """
        try:
            logger.info("Generating final validation report.")

            validation_status = all(self.critical_checks.values())
            non_critical_status = all(self.non_critical_checks.values())

            self.report.timestamp = self.timestamp
            self.report.validation_status = validation_status
            self.report.critical_passed = validation_status
            self.report.non_critical_passed = non_critical_status
            self.report.schema_check_type = self.params.schema_check.method

            if self.drift_check_performed:
                self.report.drift_check_method = self.params.drift_detection.method
            else:
                self.report.pop("drift_check_method", None)

            self.report.check_results.critical_checks = self.critical_checks.to_dict()
            self.report.check_results.non_critical_checks = self.non_critical_checks.to_dict()

            return self.report.to_dict()
        except Exception as e:
            logger.exception("Failed to generate validation report.")
            raise StudentPerformanceError(e, logger) from e

    def run_validation(self) -> DataValidationArtifact:
        """
        Executes all validation steps and returns a validation artifact.
        """
        try:
            logger.info("========== Starting Data Validation ==========")

            logger.info("Step 1: Performing schema check")
            if self.params.schema_check.method == "hash":
                self.__check_schema_hash()
            else:
                self.__check_schema_structure()

            logger.info("Step 2: Checking missing values")
            self.__check_missing_values()

            logger.info("Step 3: Checking for duplicates")
            self.__check_duplicates()

            logger.info("Step 4: Checking categorical column values")
            self.__check_categorical_values()

            if self.params.drift_detection.enabled:
                logger.info("Step 5: Performing drift check")
                self.__check_data_drift()

            logger.info("Step 6: Generating validation report")
            report = self.__generate_report()
            save_to_yaml(report, self.validation_config.validation_report_filepath, label="Validation Report")

            validation_passed = all(self.critical_checks.values())

            logger.info("Step 7: Saving validated data if validation passes")
            if validation_passed:
                save_to_csv(
                    self.df,
                    self.validation_config.validated_filepath,
                    self.validation_config.dvc_validated_filepath,
                    label="Validated Data"
                )
                logger.info("Validated data saved successfully.")
            else:
                logger.warning("Validation failed. Validated data not saved.")

            logger.info("========== Data Validation Completed ==========")

            return DataValidationArtifact(
                validated_filepath=self.validation_config.validated_filepath if validation_passed else None,
                validation_status=validation_passed
            )
        except Exception as e:
            logger.exception("Data validation process failed.")
            raise StudentPerformanceError(e, logger) from e
