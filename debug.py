from src.student_performance.pipeline.training_pipeline import TrainingPipeline
from dotenv import load_dotenv
load_dotenv()



if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
    print("shivaneee!!!")