from box import ConfigBox
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionArtifact:
    raw_artifact_path: Path
    ingested_data_filepath: Path
    dvc_raw_filepath: Path

    def __repr__(self) -> str:
        raw_artifact_str = self.raw_artifact_path.as_posix() if self.raw_artifact_path else "None"
        raw_dvc_str = self.dvc_raw_filepath.as_posix() if self.dvc_raw_filepath else "None"
        ingested_data_str = self.ingested_data_filepath.as_posix() if self.ingested_data_filepath else "None"

        return (
            "\nData Ingestion Artifact:\n"
            f"  - Raw Artifact:         '{raw_artifact_str}'\n"
            f"  - Raw DVC Path:         '{raw_dvc_str}'\n"
            f"  - Ingested Data Path:   '{ingested_data_str}'\n"
        )

@dataclass(frozen=True)
class DataValidationArtifact:
    validated_filepath: Path
    validation_status: bool

    def __repr__(self) -> str:
        validated_str = self.validated_filepath.as_posix() if self.validated_filepath else "None"

        return (
            "\nData Validation Artifact:\n"
            f"  - Validated Data Path: '{validated_str}'\n"
            f"  - Validation Status:   '{self.validation_status}'\n"
        )


@dataclass(frozen=True)
class DataTransformationArtifact:
    x_train_filepath: Path
    y_train_filepath: Path
    x_val_filepath: Path
    y_val_filepath: Path
    x_test_filepath: Path
    y_test_filepath: Path
    x_preprocessor_filepath: Path
    y_preprocessor_filepath: Path

    def __repr__(self) -> str:
        return (
            "\nData Transformation Artifact:\n"
            f"  - X Train Path:         '{self.x_train_filepath.as_posix()}'\n"
            f"  - Y Train Path:         '{self.y_train_filepath.as_posix()}'\n"
            f"  - X Val Path:           '{self.x_val_filepath.as_posix()}'\n"
            f"  - Y Val Path:           '{self.y_val_filepath.as_posix()}'\n"
            f"  - X Test Path:          '{self.x_test_filepath.as_posix()}'\n"
            f"  - Y Test Path:          '{self.y_test_filepath.as_posix()}'\n"
            f"  - X Preprocessor Path:  '{self.x_preprocessor_filepath.as_posix()}'\n"
            f"  - Y Preprocessor Path:  '{self.y_preprocessor_filepath.as_posix()}'\n"
        )
