import hashlib
import pandas as pd
from box import ConfigBox
from scipy.stats import ks_2samp
from pathlib import Path

from src.student_performance.entity.config_entity import DataValidationConfig
from src.student_performance.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger
from src.student_performance.utils.core import read_csv, save_to_csv, save_to_yaml
from src.student_performance.utils.timestamp import get_utc_timestamp
from src.student_performance.dbhandler.base_handler import DBHandler


class DataValidation:
    def __init__(
        self,
        validation_config: DataValidationConfig,
        ingestion_artifact: DataIngestionArtifact,
        backup_handler: DBHandler | None = None,
    ):
        try:
            logger.info("Initializing DataValidation component.")
            self.validation_config = validation_config
            self.schema = validation_config.schema
            self.params = validation_config.validation_params
            self.backup_handler = backup_handler
            self.timestamp = get_utc_timestamp()

            self.df = self._load_ingested_data(ingestion_artifact)
            self.base_df = None
            if (
                self.params.drift_detection.enabled
                and validation_config.dvc_validated_filepath.exists()
            ):
                self.base_df = read_csv(validation_config.dvc_validated_filepath)

            self.report = ConfigBox(validation_config.report_template.copy())
            self.critical_checks = ConfigBox({
                k: False
                for k in self.report.check_results.critical_checks.keys()
            })
            self.non_critical_checks = ConfigBox({
                k: False
                for k in self.report.check_results.non_critical_checks.keys()
            })

        except Exception as e:
            raise StudentPerformanceError(e, logger) from e

    def _load_ingested_data(
        self,
        ingestion_artifact: DataIngestionArtifact,
    ) -> pd.DataFrame:
        try:
            if ingestion_artifact.ingested_filepath:
                logger.info(
                    f"Loading ingested data from local: "
                    f"{ingestion_artifact.ingested_filepath}",
                )
                return read_csv(ingestion_artifact.ingested_filepath)

            elif ingestion_artifact.ingested_s3_uri and self.backup_handler:
                logger.info(
                    f"Loading ingested data from S3: "
                    f"{ingestion_artifact.ingested_s3_uri}"
                )
                return self.backup_handler.load_csv(
                    ingestion_artifact.ingested_s3_uri
                )

            raise ValueError("No valid ingested data source found.")
        except Exception as e:
            raise StudentPerformanceError(e, logger) from e

    def __check_schema_hash(self) -> None:
        try:
            logger.info("Performing schema hash check.")
            expected = "|".join(
                f"{col}:{dtype}"
                for col, dtype in sorted(self.schema.columns.items())
            )
            expected_hash = hashlib.md5(expected.encode()).hexdigest()

            current = "|".join(
                f"{col}:{self.df[col].dtype}"
                for col in sorted(self.df.columns)
            )
            current_hash = hashlib.md5(current.encode()).hexdigest()

            self.critical_checks.schema_is_match = (
                expected_hash == current_hash
            )
            msg = (
                "Schema hash check passed."
                if self.critical_checks.schema_is_match
                else "Schema hash mismatch."
            )
            logger.info(msg)
        except Exception as e:
            logger.exception("Schema hash check failed.")
            self.critical_checks.schema_is_match = False
            raise StudentPerformanceError(e, logger) from e

    def __check_schema_structure(self) -> None:
        try:
            logger.info("Performing schema structure check.")
            expected_cols = set(self.schema.columns.keys()) | {
                self.schema.target_column
            }
            actual_cols = set(self.df.columns)
            self.critical_checks.schema_is_match = (
                expected_cols == actual_cols
            )
            if not self.critical_checks.schema_is_match:
                logger.error(
                    f"Schema structure mismatch: "
                    f"expected={expected_cols}, actual={actual_cols}"
                )
        except Exception as e:
            logger.exception("Schema structure check failed.")
            self.critical_checks.schema_is_match = False
            raise StudentPerformanceError(e, logger) from e

    def __check_missing_values(self) -> None:
        try:
            logger.info("Checking for missing values.")
            missing = self.df.isnull().sum().to_dict()
            missing["timestamp"] = self.timestamp

            missing_report_filepath = self.validation_config.missing_report_filepath
            if self.validation_config.local_enabled:
                save_to_yaml(missing, missing_report_filepath, label="Missing Value Report")

            if self.validation_config.s3_enabled and self.backup_handler:
                missing_report_s3_key = self.validation_config.missing_report_s3_key
                with self.backup_handler as handler:
                    handler.stream_yaml(missing, missing_report_s3_key)

            self.non_critical_checks.no_missing_values = not any(
                v > 0 for v in missing.values() if isinstance(v, (int, float))
            )
        except Exception as e:
            logger.exception("Missing value check failed.")
            self.non_critical_checks.no_missing_values = False
            raise StudentPerformanceError(e, logger) from e

    def __check_duplicates(self) -> None:
        try:
            logger.info("Checking for duplicate rows.")
            before = len(self.df)
            self.df = self.df.drop_duplicates()
            removed = before - len(self.df)

            report = {"duplicate_rows_removed": removed, "timestamp": self.timestamp}

            if self.validation_config.local_enabled:
                duplicates_report_filepath = self.validation_config.duplicates_report_filepath
                save_to_yaml(report, duplicates_report_filepath, label="Duplicates Report")
            if self.validation_config.s3_enabled and self.backup_handler:
                duplicates_report_s3_key = self.validation_config.duplicates_report_s3_key
                with self.backup_handler as handler:
                    handler.stream_yaml(report, duplicates_report_s3_key)

            self.non_critical_checks.no_duplicate_rows = (removed == 0)
        except Exception as e:
            logger.exception("Duplicate check failed.")
            self.non_critical_checks.no_duplicate_rows = False
            raise StudentPerformanceError(e, logger) from e

    def __check_categorical_values(self) -> None:
        try:
            logger.info("Checking categorical values.")
            allowed = getattr(self.schema, "allowed_values", {}) or {}
            violations: dict = {}

            for col, values in allowed.items():
                if col not in self.df.columns:
                    continue
                actual = set(self.df[col].dropna().unique())
                unexpected = actual - set(values)
                if unexpected:
                    violations[col] = {
                        "unexpected_values": sorted(unexpected),
                        "expected_values": values,
                    }

            report = {
                "violations_found": bool(violations),
                "details": violations,
                "timestamp": self.timestamp,
            }

            if self.validation_config.local_enabled:
                categorical_report_filepath = self.validation_config.categorical_report_filepath
                save_to_yaml(report, categorical_report_filepath, label="Categorical Values Report")
            if self.validation_config.s3_enabled and self.backup_handler:
                categorical_report_s3_key = self.validation_config.categorical_report_s3_key
                with self.backup_handler as handler:
                    handler.stream_yaml(report, categorical_report_s3_key)

            self.non_critical_checks.categorical_values_match = not bool(violations)
        except Exception as e:
            logger.exception("Categorical value check failed.")
            self.non_critical_checks.categorical_values_match = False
            raise StudentPerformanceError(e, logger) from e

    def __check_data_drift(self) -> None:
        try:
            logger.info("Performing data drift check.")
            report: dict = {
                "drift_check_performed": bool(self.base_df is not None),
                "drift_method": self.params.drift_detection.method,
                "columns": {},
                "timestamp": self.timestamp,
            }

            if self.base_df is not None:
                drift_detected = False
                for col in self.df.columns:
                    if col in self.base_df.columns:
                        _, p = ks_2samp(self.base_df[col], self.df[col])
                        drift = p < self.params.drift_detection.p_value_threshold
                        report["columns"][col] = {"p_value": float(p), "drift": drift}
                        if drift:
                            drift_detected = True
                report["drift_detected"] = drift_detected
                self.critical_checks.no_data_drift = not drift_detected
            else:
                report["reason"] = "Base dataset not found, skipping drift."
                self.critical_checks.no_data_drift = True

            if self.validation_config.local_enabled:
                drift_report_filepath = self.validation_config.drift_report_filepath
                save_to_yaml(report, drift_report_filepath, label="Drift Report")
            if self.validation_config.s3_enabled and self.backup_handler:
                drift_report_s3_key = self.validation_config.drift_report_s3_key
                with self.backup_handler as handler:
                    handler.stream_yaml(report, drift_report_s3_key)

        except Exception as e:
            logger.exception("Data drift check failed.")
            self.critical_checks.no_data_drift = False
            raise StudentPerformanceError(e, logger) from e

    def __generate_report(self) -> dict:
        try:
            logger.info("Generating final validation report.")
            self.report.timestamp = self.timestamp
            self.report.validation_status = all(self.critical_checks.values())
            self.report.critical_passed = self.report.validation_status
            self.report.non_critical_passed = all(self.non_critical_checks.values())
            self.report.schema_check_type = self.params.schema_check.method
            if self.params.drift_detection.enabled:
                self.report.drift_check_method = self.params.drift_detection.method

            self.report.check_results.critical_checks = self.critical_checks.to_dict()
            self.report.check_results.non_critical_checks = self.non_critical_checks.to_dict()

            if self.validation_config.local_enabled:
                validation_report_filepath = self.validation_config.validation_report_filepath
                save_to_yaml(self.report, validation_report_filepath, label="Validation Report")
            if self.validation_config.s3_enabled and self.backup_handler:
                validation_report_s3_key = self.validation_config.validation_report_s3_key
                with self.backup_handler as handler:
                    handler.stream_yaml(
                        self.report,
                        validation_report_s3_key,
                    )
        except Exception as e:
            logger.exception("Failed to generate validation report.")
            raise StudentPerformanceError(e, logger) from e

    def run_validation(self) -> DataValidationArtifact:
        try:
            logger.info("========== Starting Data Validation ==========")

            if self.params.schema_check.method == "hash":
                self.__check_schema_hash()
            else:
                self.__check_schema_structure()

            self.__check_missing_values()
            self.__check_duplicates()
            self.__check_categorical_values()

            if self.params.drift_detection.enabled:
                self.__check_data_drift()

            self.__generate_report()

            passed = all(self.critical_checks.values())
            validated_local = None
            validated_s3_uri = None

            if passed:
                if self.validation_config.local_enabled:
                    save_to_csv(
                        self.df,
                        self.validation_config.validated_filepath,
                        label="Validated Data",
                    )
                    validated_local = self.validation_config.validated_filepath

                if self.validation_config.s3_enabled and self.backup_handler:
                    validated_s3_key = self.validation_config.validated_s3_key
                    with self.backup_handler as handler:
                        validated_s3_uri = handler.stream_csv(
                            self.df,
                            validated_s3_key,
                        )

            logger.info("========== Data Validation Completed ==========")
            return DataValidationArtifact(
                validated_filepath=validated_local,
                validated_s3_uri=validated_s3_uri,
                validation_status=passed,
            )

        except Exception as e:
            logger.exception("Data validation process failed.")
            raise StudentPerformanceError(e, logger) from e
