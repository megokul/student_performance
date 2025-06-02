from src.student_performance.config.configuration import ConfigurationManager
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger

class TrainingPipeline:
    def __init__(self):
        try:
            logger.info("Initializing TrainingPipeline...")
            self.config_manager = ConfigurationManager()

            # Load configuration objects
            self.postgres_config = self.config_manager.get_postgres_handler_config()
        except Exception as e:
            msg = "Failed to initialize TrainingPipeline."
            raise StudentPerformanceError(e, msg) from e


    def run_pipeline(self):
        ...