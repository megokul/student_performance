from src.student_performance.config.configuration import ConfigurationManager
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.dbhandler.postgres_dbhandler import PostgresDBHandler
from src.student_performance.dbhandler.s3_handler import S3Handler
from src.student_performance.logging import logger

from src.student_performance.components.data_ingestion import DataIngestion
from src.student_performance.components.data_validation import DataValidation
from src.student_performance.components.data_transformation import DataTransformation
from src.student_performance.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def __init__(self):
        try:
            logger.info("Initializing TrainingPipeline...")
            self.config_manager = ConfigurationManager()
        except Exception as e:
            raise StudentPerformanceError(e, "Failed to initialize TrainingPipeline.") from e

    def run_pipeline(self):
        try:
            logger.info("========== Training Pipeline Started ==========")

            # Step 1: Setup configurations and database handler
            postgres_config = self.config_manager.get_postgres_handler_config()
            s3_config = self.config_manager.get_s3_handler_config()
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            postgresdb_handler = PostgresDBHandler(config=postgres_config)
            s3_handler = S3Handler(config=s3_config)

            # Step 2: Run data ingestion
            data_ingestion = DataIngestion(
                ingestion_config=data_ingestion_config,
                source_handler=postgresdb_handler,
                backup_handler=s3_handler,
            )
            data_ingestion_artifact = data_ingestion.run_ingestion()
            logger.info(f"Data Ingestion Artifact: {data_ingestion_artifact}")

            # Step 3: Run data validation
            data_validation_config = self.config_manager.get_data_validation_config()
            data_validation = DataValidation(
                validation_config=data_validation_config,
                ingestion_artifact=data_ingestion_artifact,
                backup_handler=s3_handler,
            )
            data_validation_artifact = data_validation.run_validation()
            logger.info(f"Data Validation Artifact: {data_validation_artifact}")

            # Step 4: Run data transformation
            if data_validation_artifact.validation_status:
                data_transformation_config = self.config_manager.get_data_transformation_config()
                data_transformation = DataTransformation(
                    transformation_config=data_transformation_config,
                    validation_artifact=data_validation_artifact,
                    backup_handler=s3_handler,
                )
                data_transformation_artifact = data_transformation.run_transformation()
                logger.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            else:
                logger.warning("Data validation failed. Skipping data transformation.")
                return

            # Step 5: Run model training
            model_trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(
                config=model_trainer_config,
                transformation_artifact=data_transformation_artifact,
            )
            model_trainer_artifact = model_trainer.run_training()
            logger.info(f"Model Trainer Artifact: {model_trainer_artifact}")

            logger.info("========== Training Pipeline Completed ==========")

        except Exception as e:
            raise StudentPerformanceError(e, "TrainingPipeline failed.") from e
