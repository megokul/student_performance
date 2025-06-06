from box import ConfigBox
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PostgresDBHandlerConfig:
    root_dir: Path
    host: str
    port: int
    dbname: str
    user: str
    password: str
    table_name: str
    input_data_filepath: Path
    table_schema: ConfigBox

    def __post_init__(self):
        self.root_dir = Path(self.root_dir)
        self.input_data_filepath = Path(self.input_data_filepath)

    def __repr__(self) -> str:
        return (
            "\nPostgres DB Handler Config:\n"
            f"  - Root Dir:         '{self.root_dir.as_posix()}'\n"
            f"  - Host:             {self.host}\n"
            f"  - Port:             {self.port}\n"
            f"  - Database Name:    {self.dbname}\n"
            f"  - User:             {self.user}\n"
            f"  - Password:         {'*' * 8} (hidden)\n"
            f"  - Table:            {self.table_name}\n"
            f"  - Input Filepath:   '{self.input_data_filepath.as_posix()}'\n"
            f"  - Input Filepath:   {'table_schema'} (hidden)\n"
        )

@dataclass
class DataIngestionConfig:
    root_dir: Path
    raw_data_filepath: Path
    dvc_raw_filepath: Path
    ingested_data_filepath: Path

    def __post_init__(self):
        self.root_dir = Path(self.root_dir)
        self.raw_data_filepath = Path(self.raw_data_filepath)
        self.dvc_raw_filepath = Path(self.dvc_raw_filepath)
        self.ingested_data_filepath = Path(self.ingested_data_filepath)

@dataclass
class DataValidationConfig:
    root_dir: Path
    validated_data_filepath: Path
    dvc_validated_filepath: Path
    schema: ConfigBox
    report_template: ConfigBox
    validation_params: ConfigBox

    def __post_init__(self):
        self.root_dir = Path(self.root_dir)
        self.validated_data_filepath = Path(self.validated_data_filepath)
        self.dvc_validated_filepath = Path(self.dvc_validated_filepath)
