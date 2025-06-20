import numpy as np
import pandas as pd

from src.student_performance.entity.config_entity import DataIngestionConfig
from src.student_performance.entity.artifact_entity import DataIngestionArtifact
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger
from src.student_performance.utils.core import save_to_csv
from src.student_performance.dbhandler.base_handler import DBHandler


class DataIngestion:
    def __init__(
        self,
        ingestion_config: DataIngestionConfig,
        source_handler: DBHandler,
        backup_handler: DBHandler = None,
    ):
        try:
            self.ingestion_config = ingestion_config
            self.source_handler = source_handler
            self.backup_handler = backup_handler
        except Exception as e:
            logger.exception("Failed to initialize DataIngestion class.")
            raise StudentPerformanceError(e, logger) from e

    def __fetch_raw_data(self) -> pd.DataFrame:
        try:
            with self.source_handler as handler:
                df = handler.load_from_source()
                logger.info(f"Fetched {len(df)} raw rows from data source.")
                return df
        except Exception as e:
            logger.exception("Failed to fetch raw data from source.")
            raise StudentPerformanceError(e, logger) from e

    def __clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df_cleaned = df.drop(columns=["_id"], errors="ignore").copy()
            df_cleaned.replace({"na": np.nan}, inplace=True)
            logger.info("Raw DataFrame cleaned successfully.")
            return df_cleaned
        except Exception as e:
            logger.exception("Failed to clean raw DataFrame.")
            raise StudentPerformanceError(e, logger) from e

    def run_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("========== Starting Data Ingestion ==========")

            # Step 1: Fetch raw data
            raw_df = self.__fetch_raw_data()

            # Step 2: Clean raw data
            ingested_df = self.__clean_dataframe(raw_df)

            raw_s3_uri = dvc_raw_s3_uri = ingested_s3_uri = None

            # Step 3: Save locally if enabled
            if self.ingestion_config.local_enabled:
                logger.info("Saving raw and ingested data locally")
                save_to_csv(raw_df, self.ingestion_config.raw_filepath, label="Raw Data")
                save_to_csv(raw_df, self.ingestion_config.dvc_raw_filepath, label="Raw Data (DVC)")
                save_to_csv(ingested_df, self.ingestion_config.ingested_filepath, label="Ingested Data")

            # Step 4: Stream to S3 if enabled
            if self.ingestion_config.s3_enabled and self.backup_handler:
                logger.info("Streaming raw and ingested data to S3")
                with self.backup_handler as handler:
                    raw_s3_uri = handler.stream_csv(raw_df, self.ingestion_config.raw_filepath.as_posix())
                    dvc_raw_s3_uri = handler.stream_csv(raw_df, self.ingestion_config.dvc_raw_filepath.as_posix())
                    ingested_s3_uri = handler.stream_csv(ingested_df, self.ingestion_config.ingested_filepath.as_posix())

            logger.info("========== Data Ingestion Completed ==========")

            return DataIngestionArtifact(
                raw_filepath=self.ingestion_config.raw_filepath if self.ingestion_config.local_enabled else None,
                dvc_raw_filepath=self.ingestion_config.dvc_raw_filepath if self.ingestion_config.local_enabled else None,
                ingested_filepath=self.ingestion_config.ingested_filepath if self.ingestion_config.local_enabled else None,
                raw_s3_uri=raw_s3_uri,
                dvc_raw_s3_uri=dvc_raw_s3_uri,
                ingested_s3_uri=ingested_s3_uri,
            )

        except Exception as e:
            logger.exception("Data ingestion pipeline execution failed.")
            raise StudentPerformanceError(e, logger) from e
