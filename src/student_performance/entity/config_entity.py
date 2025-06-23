from box import ConfigBox
from dataclasses import dataclass
from pathlib import Path
from typing import List

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
class S3HandlerConfig:
    root_dir: Path
    bucket_name: str
    aws_region: str
    local_dir_to_sync: Path
    s3_artifacts_prefix: str

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.local_dir_to_sync = Path(self.local_dir_to_sync)

    def __repr__(self) -> str:
        return (
            "\nS3 Handler Config:\n",
            f"  - Root Dir:              {self.root_dir}",
            f"  - Bucket Name:           {self.bucket_name}",
            f"  - AWS Region:            {self.aws_region}",
            f"  - Local Dir to Sync:     {self.local_dir_to_sync}",
            f"  - S3 Artifacts Prefix:   {self.s3_artifacts_prefix}",
        )


@dataclass
class DataIngestionConfig:
    root_dir: Path
    raw_filepath: Path
    dvc_raw_filepath: Path
    ingested_filepath: Path

    local_enabled: bool
    s3_enabled: bool

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.raw_filepath = Path(self.raw_filepath)
        self.dvc_raw_filepath = Path(self.dvc_raw_filepath)
        self.ingested_filepath = Path(self.ingested_filepath)

    @property
    def raw_s3_key(self) -> str:
        return self.raw_filepath.as_posix()

    @property
    def dvc_raw_s3_key(self) -> str:
        return self.dvc_raw_filepath.as_posix()

    @property
    def ingested_s3_key(self) -> str:
        return self.ingested_filepath.as_posix()

    def __repr__(self) -> str:
        parts = [
            "\nData Ingestion Config:",
            f"  - Root Dir:             {self.root_dir}",
            f"  - Raw Data Path:        {self.raw_filepath}",
            f"  - DVC Raw Data Path:    {self.dvc_raw_filepath}",
            f"  - Ingested Data Path:   {self.ingested_filepath}",
            f"  - Local Save Enabled:   {self.local_enabled}",
            f"  - S3 Upload Enabled:    {self.s3_enabled}",
            f"  - Raw S3 Key:           {self.raw_s3_key}",
            f"  - DVC Raw S3 Key:       {self.dvc_raw_s3_key}",
            f"  - Ingested S3 Key:      {self.ingested_s3_key}",
        ]
        return "\n".join(parts)


@dataclass
class DataValidationConfig:
    root_dir: Path
    validated_filepath: Path
    dvc_validated_filepath: Path

    missing_report_filepath: Path
    duplicates_report_filepath: Path
    drift_report_filepath: Path
    categorical_report_filepath: Path
    validation_report_filepath: Path

    schema: ConfigBox
    report_template: ConfigBox
    validation_params: ConfigBox

    local_enabled: bool
    s3_enabled: bool

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.validated_filepath = Path(self.validated_filepath)
        self.dvc_validated_filepath = Path(self.dvc_validated_filepath)
        self.missing_report_filepath = Path(self.missing_report_filepath)
        self.duplicates_report_filepath = Path(self.duplicates_report_filepath)
        self.drift_report_filepath = Path(self.drift_report_filepath)
        self.categorical_report_filepath = Path(self.categorical_report_filepath)
        self.validation_report_filepath = Path(self.validation_report_filepath)

    @property
    def validated_s3_key(self) -> str:
        return self.validated_filepath.as_posix()

    @property
    def dvc_validated_s3_key(self) -> str:
        return self.dvc_validated_filepath.as_posix()

    @property
    def missing_report_s3_key(self) -> str:
        return self.missing_report_filepath.as_posix()

    @property
    def duplicates_report_s3_key(self) -> str:
        return self.duplicates_report_filepath.as_posix()

    @property
    def drift_report_s3_key(self) -> str:
        return self.drift_report_filepath.as_posix()

    @property
    def categorical_report_s3_key(self) -> str:
        return self.categorical_report_filepath.as_posix()

    @property
    def validation_report_s3_key(self) -> str:
        return self.validation_report_filepath.as_posix()

    def __repr__(self) -> str:
        parts = [
            "\nData Validation Config:",
            f"  - Root Dir:                   {self.root_dir}",
            f"  - Validated CSV Path:         {self.validated_filepath}",
            f"  - DVC Validated CSV Path:     {self.dvc_validated_filepath}",
            f"  - Missing Report Path:        {self.missing_report_filepath}",
            f"  - Duplicates Report Path:     {self.duplicates_report_filepath}",
            f"  - Drift Report Path:          {self.drift_report_filepath}",
            f"  - Categorical Report Path:    {self.categorical_report_filepath}",
            f"  - Validation Report Path:     {self.validation_report_filepath}",
            f"  - Local Save Enabled:         {self.local_enabled}",
            f"  - S3 Upload Enabled:          {self.s3_enabled}",
            f"  - Validated S3 Key:           {self.validated_s3_key}",
            f"  - DVC Validated S3 Key:       {self.dvc_validated_s3_key}",
            f"  - Missing Report S3 Key:      {self.missing_report_s3_key}",
            f"  - Duplicates Report S3 Key:   {self.duplicates_report_s3_key}",
            f"  - Drift Report S3 Key:        {self.drift_report_s3_key}",
            f"  - Categorical Report S3 Key:  {self.categorical_report_s3_key}",
            f"  - Validation Report S3 Key:   {self.validation_report_s3_key}",
            f"  - Schema Config:              (hidden)",
            f"  - Report Template:            (hidden)",
            f"  - Validation Params:          (hidden)",
        ]
        return "\n".join(parts)


@dataclass
class DataTransformationConfig:
    root_dir: Path

    local_enabled: bool
    s3_enabled: bool

    target_column: str
    transformation_params: ConfigBox

    x_train_filepath: Path
    y_train_filepath: Path
    x_val_filepath: Path
    y_val_filepath: Path
    x_test_filepath: Path
    y_test_filepath: Path

    x_train_dvc_filepath: Path
    y_train_dvc_filepath: Path
    x_val_dvc_filepath: Path
    y_val_dvc_filepath: Path
    x_test_dvc_filepath: Path
    y_test_dvc_filepath: Path

    x_preprocessor_filepath: Path
    y_preprocessor_filepath: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        for attr in (
            "x_train_filepath",
            "y_train_filepath",
            "x_val_filepath",
            "y_val_filepath",
            "x_test_filepath",
            "y_test_filepath",
            "x_train_dvc_filepath",
            "y_train_dvc_filepath",
            "x_val_dvc_filepath",
            "y_val_dvc_filepath",
            "x_test_dvc_filepath",
            "y_test_dvc_filepath",
            "x_preprocessor_filepath",
            "y_preprocessor_filepath",
        ):
            setattr(self, attr, Path(getattr(self, attr)))

    @property
    def x_train_s3_key(self) -> str:
        return self.x_train_filepath.as_posix()

    @property
    def y_train_s3_key(self) -> str:
        return self.y_train_filepath.as_posix()

    @property
    def x_val_s3_key(self) -> str:
        return self.x_val_filepath.as_posix()

    @property
    def y_val_s3_key(self) -> str:
        return self.y_val_filepath.as_posix()

    @property
    def x_test_s3_key(self) -> str:
        return self.x_test_filepath.as_posix()

    @property
    def y_test_s3_key(self) -> str:
        return self.y_test_filepath.as_posix()

    @property
    def x_train_dvc_s3_key(self) -> str:
        return self.x_train_dvc_filepath.as_posix()

    @property
    def y_train_dvc_s3_key(self) -> str:
        return self.y_train_dvc_filepath.as_posix()

    @property
    def x_val_dvc_s3_key(self) -> str:
        return self.x_val_dvc_filepath.as_posix()

    @property
    def y_val_dvc_s3_key(self) -> str:
        return self.y_val_dvc_filepath.as_posix()

    @property
    def x_test_dvc_s3_key(self) -> str:
        return self.x_test_dvc_filepath.as_posix()

    @property
    def y_test_dvc_s3_key(self) -> str:
        return self.y_test_dvc_filepath.as_posix()

    @property
    def x_preprocessor_s3_key(self) -> str:
        return self.x_preprocessor_filepath.as_posix()

    @property
    def y_preprocessor_s3_key(self) -> str:
        return self.y_preprocessor_filepath.as_posix()

    def __repr__(self) -> str:
        parts = [
            "\nData Transformation Config:",
            f"  - Root Dir:                  {self.root_dir}",
            f"  - Local Save Enabled:        {self.local_enabled}",
            f"  - S3 Upload Enabled:         {self.s3_enabled}",
            f"  - Target Column:             {self.target_column}",
            f"  - X Train Path:              {self.x_train_filepath}",
            f"  - Y Train Path:              {self.y_train_filepath}",
            f"  - X Val Path:                {self.x_val_filepath}",
            f"  - Y Val Path:                {self.y_val_filepath}",
            f"  - X Test Path:               {self.x_test_filepath}",
            f"  - Y Test Path:               {self.y_test_filepath}",
            f"  - X Preprocessor Path:       {self.x_preprocessor_filepath}",
            f"  - Y Preprocessor Path:       {self.y_preprocessor_filepath}",
            f"  - X Train S3 Key:            {self.x_train_s3_key}",
            f"  - Y Train S3 Key:            {self.y_train_s3_key}",
            f"  - X Val S3 Key:              {self.x_val_s3_key}",
            f"  - Y Val S3 Key:              {self.y_val_s3_key}",
            f"  - X Test S3 Key:             {self.x_test_s3_key}",
            f"  - Y Test S3 Key:             {self.y_test_s3_key}",
            f"  - X Train DVC S3 Key:        {self.x_train_dvc_s3_key}",
            f"  - Y Train DVC S3 Key:        {self.y_train_dvc_s3_key}",
            f"  - X Val DVC S3 Key:          {self.x_val_dvc_s3_key}",
            f"  - Y Val DVC S3 Key:          {self.y_val_dvc_s3_key}",
            f"  - X Test DVC S3 Key:         {self.x_test_dvc_s3_key}",
            f"  - Y Test DVC S3 Key:         {self.y_test_dvc_s3_key}",
            f"  - X Preprocessor S3 Key:     {self.x_preprocessor_s3_key}",
            f"  - Y Preprocessor S3 Key:     {self.y_preprocessor_s3_key}",
            f"  - Transformation Params:     (hidden)",
            f"  - DVC-tracked Filepaths:     (hidden)",
        ]
        return "\n".join(parts)


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    trained_model_filepath: Path
    training_report_filepath: Path
    inference_model_filepath: Path
    inference_model_serving_filepath: Path

    local_enabled: bool
    s3_enabled: bool

    models: List[dict]
    optimization: ConfigBox
    tracking: ConfigBox

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.trained_model_filepath = Path(self.trained_model_filepath)
        self.training_report_filepath = Path(self.training_report_filepath)

    @property
    def trained_model_s3_key(self) -> str:
        return self.trained_model_filepath.as_posix()

    @property
    def training_report_s3_key(self) -> str:
        return self.training_report_filepath.as_posix()

    @property
    def inference_model_s3_key(self) -> str:
        return self.inference_model_filepath.as_posix()

    def __repr__(self) -> str:
        parts = [
            "\nModel Trainer Config:",
            f"  - Root Dir:                  {self.root_dir}",
            f"  - Trained Model Path:        {self.trained_model_filepath}",
            f"  - Training Report Path:      {self.training_report_filepath}",
            f"  - Local Save Enabled:        {self.local_enabled}",
            f"  - S3 Upload Enabled:         {self.s3_enabled}",
            f"  - Trained Model S3 Key:      {self.trained_model_s3_key}",
            f"  - Training Report S3 Key:    {self.training_report_s3_key}",
            f"  - Models:                    (hidden)",
            f"  - Optimization:              (hidden)",
            f"  - Tracking:                  (hidden)",
        ]
        return "\n".join(parts)
