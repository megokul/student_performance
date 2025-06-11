import numpy as np
import pandas as pd
from src.student_performance.entity.config_entity import DataIngestionConfig
from src.student_performance.entity.artifact_entity import DataIngestionArtifact
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger
from src.student_performance.utils.core import save_to_csv
from src.student_performance.dbhandler.base_handler import DBHandler


class DataIngestion:
    def __init__(self, ingestion_config: DataIngestionConfig, db_handler: DBHandler):
        try:
            self.ingestion_config = ingestion_config
            self.db_handler = db_handler
        except Exception as e:
            logger.exception("Failed to initialize DataIngestion class.")
            raise StudentPerformanceError(e, logger) from e

    def __fetch_raw_data(self) -> pd.DataFrame:
        try:
            with self.db_handler as handler:
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
        """Runs the data ingestion process."""
        try:
            logger.info("========== Starting Data Ingestion ==========")

            # Step 1: Fetch raw data
            logger.info("Step 1: Fetching raw data")
            raw_df = self.__fetch_raw_data()

            # Step 2: Save raw data
            logger.info("Step 2: Saving raw data")
            save_to_csv(
                raw_df,
                self.ingestion_config.raw_data_filepath,
                label="Raw Data",
            )

            # Step 3: Save raw data to dvc path
            logger.info("Step 3: Saving raw data to dvc path")
            save_to_csv(
                raw_df,
                self.ingestion_config.dvc_raw_filepath,
                label="Raw Data",
            )

            # Step 4: Clean raw data
            logger.info("Step 4: Cleaning raw data")
            cleaned_df = self.__clean_dataframe(raw_df)

            # Step 5: Save cleaned (ingested) data
            logger.info("Step 5: Saving cleaned (ingested) data")
            save_to_csv(
                cleaned_df,
                self.ingestion_config.ingested_data_filepath,
                label="Ingested Data",
            )

            logger.info("========== Data Ingestion Completed ==========")

            return DataIngestionArtifact(
                raw_artifact_path=self.ingestion_config.raw_data_filepath,
                ingested_data_filepath=self.ingestion_config.ingested_data_filepath,
                dvc_raw_filepath=self.ingestion_config.dvc_raw_filepath,
            )
        except Exception as e:
            logger.exception("Data ingestion pipeline execution failed.")
            raise StudentPerformanceError(e, logger) from e
