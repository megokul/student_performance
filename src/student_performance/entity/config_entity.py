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

    def __repr__(self) -> str:
        return (
            "\nData Ingestion Config:\n"
            f"  - Root Dir:           '{self.root_dir.as_posix()}'\n"
            f"  - Raw Data Path:      '{self.raw_data_filepath.as_posix()}'\n"
            f"  - DVC Raw Path:       '{self.dvc_raw_filepath.as_posix()}'\n"
            f"  - Ingested Data Path: '{self.ingested_data_filepath.as_posix()}'\n"
        )


@dataclass
class DataValidationConfig:
    root_dir: Path
    validated_filepath: Path
    dvc_validated_filepath: Path

    schema: ConfigBox
    report_template: ConfigBox
    validation_params: ConfigBox

    missing_report_filepath: Path
    duplicates_report_filepath: Path
    drift_report_filepath: Path
    validation_report_filepath: Path
    categorical_report_filepath: Path

    def __post_init__(self):
        self.root_dir = Path(self.root_dir)
        self.validated_filepath = Path(self.validated_filepath)
        self.dvc_validated_filepath = Path(self.dvc_validated_filepath)

        self.missing_report_filepath = Path(self.missing_report_filepath)
        self.duplicates_report_filepath = Path(self.duplicates_report_filepath)
        self.drift_report_filepath = Path(self.drift_report_filepath)
        self.validation_report_filepath = Path(self.validation_report_filepath)
        self.categorical_report_filepath = Path(self.categorical_report_filepath)

    def __repr__(self) -> str:
        return (
            "\nData Validation Config:\n"
            f"  - Root Dir:                '{self.root_dir.as_posix()}'\n"
            f"  - Validated CSV:           '{self.validated_filepath.as_posix()}'\n"
            f"  - DVC Validated Path:      '{self.dvc_validated_filepath.as_posix()}'\n"
            f"  - Missing Report:          '{self.missing_report_filepath.as_posix()}'\n"
            f"  - Duplicates Report:       '{self.duplicates_report_filepath.as_posix()}'\n"
            f"  - Drift Report:            '{self.drift_report_filepath.as_posix()}'\n"
            f"  - Categorical Report:      '{self.categorical_report_filepath.as_posix()}'\n"
            f"  - Validation Report:       '{self.validation_report_filepath.as_posix()}'\n"
            f"  - Schema Config:           'schema' (hidden)\n"
            f"  - Report Template:         'template' (hidden)\n"
            f"  - Validation Params:       'params' (hidden)\n"
        )


@dataclass
class DataTransformationConfig:
    root_dir: Path

    # Target and parameters
    target_column: str
    transformation_params: ConfigBox

    # Transformed dataset filepaths
    x_train_filepath: Path
    y_train_filepath: Path
    x_val_filepath: Path
    y_val_filepath: Path
    x_test_filepath: Path
    y_test_filepath: Path

    # DVC-tracked filepaths
    x_train_dvc_filepath: Path
    y_train_dvc_filepath: Path
    x_val_dvc_filepath: Path
    y_val_dvc_filepath: Path
    x_test_dvc_filepath: Path
    y_test_dvc_filepath: Path

    # Preprocessor objects
    x_preprocessor_filepath: Path
    y_preprocessor_filepath: Path

    def __post_init__(self):
        self.root_dir = Path(self.root_dir)
        self.x_train_filepath = Path(self.x_train_filepath)
        self.y_train_filepath = Path(self.y_train_filepath)
        self.x_val_filepath = Path(self.x_val_filepath)
        self.y_val_filepath = Path(self.y_val_filepath)
        self.x_test_filepath = Path(self.x_test_filepath)
        self.y_test_filepath = Path(self.y_test_filepath)

        self.x_train_dvc_filepath = Path(self.x_train_dvc_filepath)
        self.y_train_dvc_filepath = Path(self.y_train_dvc_filepath)
        self.x_val_dvc_filepath = Path(self.x_val_dvc_filepath)
        self.y_val_dvc_filepath = Path(self.y_val_dvc_filepath)
        self.x_test_dvc_filepath = Path(self.x_test_dvc_filepath)
        self.y_test_dvc_filepath = Path(self.y_test_dvc_filepath)

        self.x_preprocessor_filepath = Path(self.x_preprocessor_filepath)
        self.y_preprocessor_filepath = Path(self.y_preprocessor_filepath)

    def __repr__(self) -> str:
        return (
            "\nData Transformation Config:\n"
            f"  - Root Dir:              '{self.root_dir.as_posix()}'\n"
            f"  - Target Column:         '{self.target_column}'\n"
            f"  - X Train:               '{self.x_train_filepath.as_posix()}'\n"
            f"  - Y Train:               '{self.y_train_filepath.as_posix()}'\n"
            f"  - X Val:                 '{self.x_val_filepath.as_posix()}'\n"
            f"  - Y Val:                 '{self.y_val_filepath.as_posix()}'\n"
            f"  - X Test:                '{self.x_test_filepath.as_posix()}'\n"
            f"  - Y Test:                '{self.y_test_filepath.as_posix()}'\n"
            f"  - X Preprocessor:        '{self.x_preprocessor_filepath.as_posix()}'\n"
            f"  - Y Preprocessor:        '{self.y_preprocessor_filepath.as_posix()}'\n"
            f"  - Transformation Params: 'transformation_params' (hidden)\n"
        )
