from src.student_performance.constants.constants import (
    CONFIG_DIR,
    CONFIG_FILENAME,
    PARAMS_FILENAME,
    SCHEMA_FILENAME,
    TEMPLATES_FILENAME,
    LOGS_ROOT,
    ARTIFACTS_ROOT,
    POSTGRES_HANDLER_ROOT,
)
from pathlib import Path
import os
from src.student_performance.utils.timestamp import get_utc_timestamp
from src.student_performance.utils.core import read_yaml
from src.student_performance.entity.config_entity import (
    PostgresDBHandlerConfig,
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
        config_dir = Path(CONFIG_DIR)
        config_filepath = config_dir / CONFIG_FILENAME
        params_filepath = config_dir / PARAMS_FILENAME
        schema_filepath = config_dir / SCHEMA_FILENAME
        templates_filepath = config_dir / TEMPLATES_FILENAME

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
