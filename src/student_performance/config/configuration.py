from src.student_performance.constants.constants import (
    CONFIG_ROOT,
    CONFIG_FILENAME,
    PARAMS_FILENAME,
    SCHEMA_FILENAME,
    TEMPLATES_FILENAME,
    LOGS_ROOT,
    ARTIFACTS_ROOT,
    POSTGRES_HANDLER_ROOT,
    DVC_ROOT,
    DVC_RAW_SUBDIR,
    DVC_VALIDATED_SUBDIR,
    DVC_TRANSFORMED_SUBDIR,
    INGEST_ROOT,
    INGEST_RAW_SUBDIR,
    INGEST_INGESTED_SUBDIR,
    VALID_ROOT,
    VALID_VALIDATED_SUBDIR,
    VALID_REPORTS_SUBDIR,
    TRANSFORM_ROOT,
    TRANSFORM_TRAIN_SUBDIR,
    TRANSFORM_TEST_SUBDIR,
    TRANSFORM_VAL_SUBDIR,
    TRANSFORM_PROCESSOR_SUBDIR,
    TRAINER_ROOT,
    TRAINER_MODEL_SUBDIR,
    TRAINER_REPORTS_SUBDIR,
    INFERENCE_MODEL_ROOT,
    TRAINER_INFERENCE_SUBDIR,
)

from pathlib import Path
import os
from src.student_performance.utils.timestamp import get_utc_timestamp
from src.student_performance.utils.core import read_yaml
from src.student_performance.entity.config_entity import (
    PostgresDBHandlerConfig,
    S3HandlerConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

class ConfigurationManager:
    _global_timestamp: str = None
    def __init__(self) -> None:
        self._init_artifacts()
        self._load_configs()

    def _init_artifacts(self) -> None:
        if ConfigurationManager._global_timestamp is None:
            ConfigurationManager._global_timestamp = get_utc_timestamp()

        timestamp = ConfigurationManager._global_timestamp
        self.artifacts_root = Path(ARTIFACTS_ROOT) / timestamp
        self.logs_root = Path(LOGS_ROOT) / timestamp

    def _load_configs(self) -> None:
        config_root = Path(CONFIG_ROOT)
        config_filepath = config_root / CONFIG_FILENAME
        params_filepath = config_root / PARAMS_FILENAME
        schema_filepath = config_root / SCHEMA_FILENAME
        templates_filepath = config_root / TEMPLATES_FILENAME

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        self.templates = read_yaml(templates_filepath)

    def get_postgres_handler_config(self) -> PostgresDBHandlerConfig:

        postgres_config = self.config.postgres_dbhandler
        root_dir = self.artifacts_root / POSTGRES_HANDLER_ROOT
        input_data_filepath = Path(postgres_config.input_data_dir) / postgres_config.input_data_filename
        table_schema = self.schema.table_schema

        return PostgresDBHandlerConfig(
            root_dir=root_dir,
            host=os.getenv("RDS_HOST"),
            port=os.getenv("RDS_PORT"),
            dbname=postgres_config.dbname,
            user=os.getenv("RDS_USER"),
            password=os.getenv("RDS_PASS"),
            table_name=postgres_config.table_name,
            input_data_filepath=input_data_filepath,
            table_schema=table_schema,
        )

    def get_s3_handler_config(self) -> S3HandlerConfig:
        s3_config = self.config.s3_handler  # your config.yaml has these under model_pusher
        root_dir = self.artifacts_root / "s3_handler"
        aws_region = os.getenv("AWS_REGION")

        return S3HandlerConfig(
            root_dir=root_dir,
            bucket_name=s3_config.s3_bucket,
            aws_region=aws_region,
            local_dir_to_sync=self.artifacts_root,  # assuming you want to sync entire artifacts dir
            s3_artifacts_prefix=s3_config.s3_artifacts_prefix,
        )

    def _build_s3_key(self, prefix: str, path: Path, relative_to: Path) -> str:
        """
        Build an S3 key by combining a prefix and the relative path of a file.

        Args:
            prefix (str): S3 base prefix (e.g., 'artifacts')
            path (Path): Full local path to the file
            relative_to (Path): The root to compute relative path from

        Returns:
            str: S3 key in POSIX format (forward slashes)
        """
        return f"{prefix}/{path.relative_to(relative_to).as_posix()}"


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        ingestion_config = self.config.data_ingestion
        data_backup_config = self.config.data_backup

        # File names
        raw_name = ingestion_config.raw_data_filename
        ingested_name = ingestion_config.ingested_data_filename

        # Local paths
        root_dir = self.artifacts_root / INGEST_ROOT
        raw_filepath = root_dir / INGEST_RAW_SUBDIR / raw_name
        dvc_raw_filepath = Path(DVC_ROOT) / DVC_RAW_SUBDIR / raw_name
        ingested_filepath = root_dir / INGEST_INGESTED_SUBDIR / ingested_name

        return DataIngestionConfig(
            root_dir=root_dir,
            raw_filepath=raw_filepath,
            dvc_raw_filepath=dvc_raw_filepath,
            ingested_filepath=ingested_filepath,
            local_enabled=data_backup_config.local_enabled,
            s3_enabled=data_backup_config.s3_enabled,
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        validation_config = self.config.data_validation
        data_backup_config = self.config.data_backup
        schema = self.schema.validation_schema
        report_template = self.templates.validation_report
        validation_params = self.params.validation_params

        validated_data_filename = validation_config.validated_data_filename
        missing_report_filename = validation_config.missing_report_filename
        duplicates_report_filename = validation_config.duplicates_report_filename
        drift_report_filename = validation_config.drift_report_filename
        validation_report_filename = validation_config.validation_report_filename
        categorical_report_filename = validation_config.categorical_report_filename

        root_dir = self.artifacts_root / VALID_ROOT
        validated_filepath = root_dir / VALID_VALIDATED_SUBDIR / validated_data_filename
        missing_report_filepath = root_dir / VALID_REPORTS_SUBDIR / missing_report_filename
        duplicates_report_filepath = root_dir / VALID_REPORTS_SUBDIR / duplicates_report_filename
        drift_report_filepath = root_dir / VALID_REPORTS_SUBDIR / drift_report_filename
        validation_report_filepath = root_dir / VALID_REPORTS_SUBDIR / validation_report_filename
        categorical_report_filepath =  root_dir / VALID_REPORTS_SUBDIR / categorical_report_filename

        dvc_validated_filepath = Path(DVC_ROOT) / DVC_VALIDATED_SUBDIR / validated_data_filename

        return DataValidationConfig(
            root_dir=root_dir,
            validated_filepath=validated_filepath,
            dvc_validated_filepath=dvc_validated_filepath,
            schema=schema,
            report_template=report_template,
            validation_params=validation_params,
            missing_report_filepath=missing_report_filepath,
            duplicates_report_filepath=duplicates_report_filepath,
            drift_report_filepath=drift_report_filepath,
            validation_report_filepath=validation_report_filepath,
            categorical_report_filepath=categorical_report_filepath,
            local_enabled=data_backup_config.local_enabled,
            s3_enabled=data_backup_config.s3_enabled,
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        transformation_config = self.config.data_transformation
        transformation_params = self.params.transformation_params
        output_column = self.schema.target_column
        data_backup_config = self.config.data_backup

        root_dir = self.artifacts_root / TRANSFORM_ROOT

        # Local paths
        x_train = root_dir / TRANSFORM_TRAIN_SUBDIR / transformation_config.x_train_filename
        y_train = root_dir / TRANSFORM_TRAIN_SUBDIR / transformation_config.y_train_filename
        x_val = root_dir / TRANSFORM_VAL_SUBDIR / transformation_config.x_val_filename
        y_val = root_dir / TRANSFORM_VAL_SUBDIR / transformation_config.y_val_filename
        x_test = root_dir / TRANSFORM_TEST_SUBDIR / transformation_config.x_test_filename
        y_test = root_dir / TRANSFORM_TEST_SUBDIR / transformation_config.y_test_filename

        # DVC-tracked paths
        dvc_root = Path(DVC_ROOT) / DVC_TRANSFORMED_SUBDIR

        x_train_dvc = dvc_root / transformation_config.x_train_filename
        y_train_dvc = dvc_root / transformation_config.y_train_filename
        x_val_dvc = dvc_root / transformation_config.x_val_filename
        y_val_dvc = dvc_root / transformation_config.y_val_filename
        x_test_dvc = dvc_root / transformation_config.x_test_filename
        y_test_dvc = dvc_root / transformation_config.y_test_filename

        # Preprocessor objects
        x_processor_path = root_dir / TRANSFORM_PROCESSOR_SUBDIR / transformation_config.x_preprocessor_filename
        y_processor_path = root_dir / TRANSFORM_PROCESSOR_SUBDIR / transformation_config.y_preprocessor_filename

        return DataTransformationConfig(
            root_dir=root_dir,
            target_column=output_column,
            transformation_params=transformation_params,
            x_train_filepath=x_train,
            y_train_filepath=y_train,
            x_val_filepath=x_val,
            y_val_filepath=y_val,
            x_test_filepath=x_test,
            y_test_filepath=y_test,
            x_train_dvc_filepath=x_train_dvc,
            y_train_dvc_filepath=y_train_dvc,
            x_val_dvc_filepath=x_val_dvc,
            y_val_dvc_filepath=y_val_dvc,
            x_test_dvc_filepath=x_test_dvc,
            y_test_dvc_filepath=y_test_dvc,
            x_preprocessor_filepath=x_processor_path,
            y_preprocessor_filepath=y_processor_path,
            local_enabled=data_backup_config.local_enabled,
            s3_enabled=data_backup_config.s3_enabled,
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        trainer_config = self.config.model_trainer
        trainer_params = self.params.model_trainer
        data_backup_config = self.config.data_backup

        root_dir = self.artifacts_root / TRAINER_ROOT
        inference_model_filepath = Path(INFERENCE_MODEL_ROOT) / trainer_config.inference_model_filename
        inference_model_serving_filepath = root_dir / TRAINER_INFERENCE_SUBDIR / trainer_config.inference_model_filename
        trained_model_filepath = root_dir / TRAINER_MODEL_SUBDIR / trainer_config.trained_model_filename
        training_report_filepath = root_dir / TRAINER_REPORTS_SUBDIR / trainer_config.training_report_filename

        mlflow_cfg = trainer_params.tracking
        mlflow_cfg.tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        return ModelTrainerConfig(
            root_dir=root_dir,
            trained_model_filepath=trained_model_filepath,
            training_report_filepath=training_report_filepath,
            models=trainer_params.models,
            optimization=trainer_params.optimization,
            tracking=mlflow_cfg,
            local_enabled=data_backup_config.local_enabled,
            s3_enabled=data_backup_config.s3_enabled,
            inference_model_filepath=inference_model_filepath,
            inference_model_serving_filepath=inference_model_serving_filepath,
        )