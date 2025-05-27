
from src.student_performance.constants.constants import (CONFIG_FILEPATH,
                                                         PARAMS_FILEPATH,
                                                         SCHEMA_FILEPATH,
                                                         TEMPLATES_FILEPATH,
                                                         )
from pathlib import Path
from src.student_performance.utils.timestamp import get_utc_timestamp
from src.student_performance.utils.core import read_yaml

class ConfigurationManager:
    _global_timestamp: str = None
    def __init__(self,
                 config_filepath: Path = CONFIG_FILEPATH,
                 params_filepath: Path = PARAMS_FILEPATH,
                 schema_filepath: Path = SCHEMA_FILEPATH,
                 templates_filepath: Path = TEMPLATES_FILEPATH,
                 ) -> None:
        self._initialize()
        self._load_configs(config_filepath, params_filepath, schema_filepath, templates_filepath)

    def _initialize(self) -> None:
        if ConfigurationManager._global_timestamp is None:
            ConfigurationManager._global_timestamp = get_utc_timestamp

    def _load_configs(self, 
                      config_filepath: Path, 
                      params_filepath: Path, 
                      schema_filepath: Path, 
                      templates_filepath: Path,
                      ):
        self.config_filepath = read_yaml(config_filepath)
        self.params_filepath = read_yaml(params_filepath)
        self.schema_filepath = read_yaml(schema_filepath)
        self.templates_filepath = read_yaml(templates_filepath)

