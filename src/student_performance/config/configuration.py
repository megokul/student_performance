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
    VALIDATION_ROOT,
    VALIDATION_SUBDIR,
)
from pathlib import Path
import os
from src.student_performance.utils.timestamp import get_utc_timestamp
from src.student_performance.utils.core import read_yaml
from src.student_performance.entity.config_entity import (
    PostgresDBHandlerConfig,
    DataIngestionConfig,
    DataValidationConfig,
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

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        ingestion_config = self.config.data_ingestion
        raw_data_filename = ingestion_config.raw_data_filename
        ingested_data_filename = ingestion_config.ingested_data_filename

        root_dir = self.artifacts_root / INGEST_ROOT
        raw_data_filepath = root_dir / INGEST_RAW_SUBDIR / raw_data_filename
        dvc_raw_filepath = Path(DVC_ROOT) / DVC_RAW_SUBDIR / raw_data_filename
        ingested_data_filepath = root_dir / INGEST_INGESTED_SUBDIR / ingested_data_filename

        return DataIngestionConfig(
            root_dir=root_dir,
            raw_data_filepath=raw_data_filepath,
            dvc_raw_filepath=dvc_raw_filepath,
            ingested_data_filepath=ingested_data_filepath,
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        validation_config = self.config.data_validation
        validated_data_filename = validation_config.validated_data_filename
        schema = self.schema.validation_schema
        report_template = self.templates.validation_report
        validation_params = self.params.validation_params

        root_dir = self.artifacts_root / VALIDATION_ROOT
        validated_data_filepath = root_dir / VALIDATION_SUBDIR / validated_data_filename
        dvc_validated_filepath = Path(DVC_ROOT) / DVC_VALIDATED_SUBDIR / validated_data_filename

        return DataValidationConfig(
            root_dir=root_dir,
            validated_data_filepath=validated_data_filepath,
            dvc_validated_filepath=dvc_validated_filepath,
            schema=schema,
            report_template=report_template,
            validation_params=validation_params,
        )
