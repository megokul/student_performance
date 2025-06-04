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