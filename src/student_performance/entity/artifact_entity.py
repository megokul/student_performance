from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionArtifact:
    raw_filepath: Path | None = None
    dvc_raw_filepath: Path | None = None
    ingested_filepath: Path | None = None
    raw_s3_uri: str | None = None
    dvc_raw_s3_uri: str | None = None
    ingested_s3_uri: str | None = None

    def __repr__(self) -> str:
        raw_local_str = self.raw_filepath.as_posix() if self.raw_filepath else "None"
        dvc_raw_local_str = self.dvc_raw_filepath.as_posix() if self.dvc_raw_filepath else "None"
        ingested_local_str = self.ingested_filepath.as_posix() if self.ingested_filepath else "None"

        raw_s3_str = self.raw_s3_uri if self.raw_s3_uri else "None"
        dvc_raw_s3_str = self.dvc_raw_s3_uri if self.dvc_raw_s3_uri else "None"
        ingested_s3_str = self.ingested_s3_uri if self.ingested_s3_uri else "None"

        return (
            "\nData Ingestion Artifact:\n"
            f"  - Raw Local Path:        '{raw_local_str}'\n"
            f"  - DVC Raw Local Path:    '{dvc_raw_local_str}'\n"
            f"  - Ingested Local Path:   '{ingested_local_str}'\n"
            f"  - Raw S3 URI:            '{raw_s3_str}'\n"
            f"  - DVC Raw S3 URI:        '{dvc_raw_s3_str}'\n"
            f"  - Ingested S3 URI:       '{ingested_s3_str}'"
        )


@dataclass(frozen=True)
class DataValidationArtifact:
    validated_filepath: Path | None = None
    validated_s3_uri: str | None = None
    validation_status: bool = False

    def __repr__(self) -> str:
        validated_local = self.validated_filepath.as_posix() if self.validated_filepath else "None"
        validated_s3 = self.validated_s3_uri if self.validated_s3_uri else "None"

        return (
            "\nData Validation Artifact:\n"
            f"  - Validated Local Path: '{validated_local}'\n"
            f"  - Validated S3 URI:     '{validated_s3}'\n"
            f"  - Validation Status:    '{self.validation_status}'"
        )


@dataclass(frozen=True)
class DataTransformationArtifact:
    x_train_filepath: Path | None = None
    y_train_filepath: Path | None = None
    x_val_filepath: Path | None = None
    y_val_filepath: Path | None = None
    x_test_filepath: Path | None = None
    y_test_filepath: Path | None = None
    x_preprocessor_filepath: Path | None = None
    y_preprocessor_filepath: Path | None = None

    x_train_s3_uri: str | None = None
    y_train_s3_uri: str | None = None
    x_val_s3_uri: str | None = None
    y_val_s3_uri: str | None = None
    x_test_s3_uri: str | None = None
    y_test_s3_uri: str | None = None
    x_preprocessor_s3_uri: str | None = None
    y_preprocessor_s3_uri: str | None = None

    def __repr__(self) -> str:
        xt_local = (
            self.x_train_filepath.as_posix()
            if self.x_train_filepath
            else "None"
        )
        yt_local = (
            self.y_train_filepath.as_posix()
            if self.y_train_filepath
            else "None"
        )
        xv_local = (
            self.x_val_filepath.as_posix()
            if self.x_val_filepath
            else "None"
        )
        yv_local = (
            self.y_val_filepath.as_posix()
            if self.y_val_filepath
            else "None"
        )
        xts_local = (
            self.x_test_filepath.as_posix()
            if self.x_test_filepath
            else "None"
        )
        yts_local = (
            self.y_test_filepath.as_posix()
            if self.y_test_filepath
            else "None"
        )
        xp_local = (
            self.x_preprocessor_filepath.as_posix()
            if self.x_preprocessor_filepath
            else "None"
        )
        yp_local = (
            self.y_preprocessor_filepath.as_posix()
            if self.y_preprocessor_filepath
            else "None"
        )

        xt_s3 = self.x_train_s3_uri or "None"
        yt_s3 = self.y_train_s3_uri or "None"
        xv_s3 = self.x_val_s3_uri or "None"
        yv_s3 = self.y_val_s3_uri or "None"
        xts_s3 = self.x_test_s3_uri or "None"
        yts_s3 = self.y_test_s3_uri or "None"
        xp_s3 = self.x_preprocessor_s3_uri or "None"
        yp_s3 = self.y_preprocessor_s3_uri or "None"

        return (
            "\nData Transformation Artifact:\n"
            f"  - X Train Local Path:        '{xt_local}'\n"
            f"  - Y Train Local Path:        '{yt_local}'\n"
            f"  - X Val Local Path:          '{xv_local}'\n"
            f"  - Y Val Local Path:          '{yv_local}'\n"
            f"  - X Test Local Path:         '{xts_local}'\n"
            f"  - Y Test Local Path:         '{yts_local}'\n"
            f"  - X Preprocessor Local Path: '{xp_local}'\n"
            f"  - Y Preprocessor Local Path: '{yp_local}'\n"
            f"  - X Train S3 URI:            '{xt_s3}'\n"
            f"  - Y Train S3 URI:            '{yt_s3}'\n"
            f"  - X Val S3 URI:              '{xv_s3}'\n"
            f"  - Y Val S3 URI:              '{yv_s3}'\n"
            f"  - X Test S3 URI:             '{xts_s3}'\n"
            f"  - Y Test S3 URI:             '{yts_s3}'\n"
            f"  - X Preprocessor S3 URI:     '{xp_s3}'\n"
            f"  - Y Preprocessor S3 URI:     '{yp_s3}'"
        )


@dataclass(frozen=True)
class ModelTrainerArtifact:
    trained_model_filepath: Path | None = None
    training_report_filepath: Path | None = None

    x_train_filepath: Path | None = None
    y_train_filepath: Path | None = None
    x_val_filepath: Path | None = None
    y_val_filepath: Path | None = None
    x_test_filepath: Path | None = None
    y_test_filepath: Path | None = None

    trained_model_s3_uri: str | None = None
    training_report_s3_uri: str | None = None

    x_train_s3_uri: str | None = None
    y_train_s3_uri: str | None = None
    x_val_s3_uri: str | None = None
    y_val_s3_uri: str | None = None
    x_test_s3_uri: str | None = None
    y_test_s3_uri: str | None = None

    def __repr__(self) -> str:
        tm_local = (
            self.trained_model_filepath.as_posix()
            if self.trained_model_filepath
            else "None"
        )
        tr_local = (
            self.training_report_filepath.as_posix()
            if self.training_report_filepath
            else "None"
        )

        xt_local = (
            self.x_train_filepath.as_posix()
            if self.x_train_filepath
            else "None"
        )
        yt_local = (
            self.y_train_filepath.as_posix()
            if self.y_train_filepath
            else "None"
        )
        xv_local = (
            self.x_val_filepath.as_posix()
            if self.x_val_filepath
            else "None"
        )
        yv_local = (
            self.y_val_filepath.as_posix()
            if self.y_val_filepath
            else "None"
        )
        xts_local = (
            self.x_test_filepath.as_posix()
            if self.x_test_filepath
            else "None"
        )
        yts_local = (
            self.y_test_filepath.as_posix()
            if self.y_test_filepath
            else "None"
        )

        tm_s3 = self.trained_model_s3_uri or "None"
        tr_s3 = self.training_report_s3_uri or "None"

        xt_s3 = self.x_train_s3_uri or "None"
        yt_s3 = self.y_train_s3_uri or "None"
        xv_s3 = self.x_val_s3_uri or "None"
        yv_s3 = self.y_val_s3_uri or "None"
        xts_s3 = self.x_test_s3_uri or "None"
        yts_s3 = self.y_test_s3_uri or "None"

        return (
            "\nModel Trainer Artifact:\n"
            f"  - Trained Model Local Path:      '{tm_local}'\n"
            f"  - Training Report Local Path:    '{tr_local}'\n"
            "\n"
            f"  - X Train Local Path:            '{xt_local}'\n"
            f"  - Y Train Local Path:            '{yt_local}'\n"
            f"  - X Val Local Path:              '{xv_local}'\n"
            f"  - Y Val Local Path:              '{yv_local}'\n"
            f"  - X Test Local Path:             '{xts_local}'\n"
            f"  - Y Test Local Path:             '{yts_local}'\n"
            "\n"
            f"  - Trained Model S3 URI:          '{tm_s3}'\n"
            f"  - Training Report S3 URI:        '{tr_s3}'\n"
            "\n"
            f"  - X Train S3 URI:                '{xt_s3}'\n"
            f"  - Y Train S3 URI:                '{yt_s3}'\n"
            f"  - X Val S3 URI:                  '{xv_s3}'\n"
            f"  - Y Val S3 URI:                  '{yv_s3}'\n"
            f"  - X Test S3 URI:                 '{xts_s3}'\n"
            f"  - Y Test S3 URI:                 '{yts_s3}'"
        )

