from src.student_performance.config.configuration import ConfigurationManager
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.dbhandler.postgres_dbhandler import PostgresDBHandler
from src.student_performance.logging import logger

from src.student_performance.components.data_ingestion import DataIngestion
from src.student_performance.components.data_validation import DataValidation


class TrainingPipeline:
    def __init__(self):
        try:
            logger.info("Initializing TrainingPipeline...")
            self.config_manager = ConfigurationManager()

        except Exception as e:
            msg = "Failed to initialize TrainingPipeline."
            raise StudentPerformanceError(e, msg) from e

    def run_pipeline(self):
        try:
            logger.info("========== Training Pipeline Started ==========")

            # Step 1: Setup configurations and database handler
            postgres_config = self.config_manager.get_postgres_handler_config()
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            postgresdb_handler = PostgresDBHandler(postgres_config=postgres_config)

            # Step 2: Run data ingestion
            data_ingestion = DataIngestion(
                ingestion_config=data_ingestion_config,
                db_handler=postgresdb_handler,
            )
            data_ingestion_artifact = data_ingestion.run_ingestion()
            logger.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")

            # Step 3: Run data validation
            data_validation_config = self.config_manager.get_data_validation_config()
            data_validation = DataValidation(
                validation_config=data_validation_config,
                ingestion_artifact=data_ingestion_artifact,
            )
            data_validation_artifact = data_validation.run_validation()
            logger.info(f"Data Validation Artifact: {data_validation_artifact}")

            logger.info("========== Training Pipeline Completed ==========")

        except Exception as e:
            msg = "TrainingPipeline failed."
            raise StudentPerformanceError(e, msg) from e
