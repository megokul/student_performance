from flask import Flask, request, jsonify, render_template
from pathlib import Path
import numpy as np
import pandas as pd
from src.student_performance.config.configuration import ConfigurationManager
from src.student_performance.components.model_prediction import ModelPrediction
from src.student_performance.dbhandler.s3_handler import S3Handler
from src.student_performance.exception.exception import StudentPerformanceError
from src.student_performance.logging import logger

app = Flask(__name__)

# Initialize once at app startup
try:
    config_manager = ConfigurationManager()
    prediction_config = config_manager.get_model_prediction_config()
    s3_handler = S3Handler(config_manager.get_s3_handler_config()) if prediction_config.s3_enabled else None
    predictor = ModelPrediction(prediction_config, backup_handler=s3_handler)
except Exception as e:
    logger.exception("Failed to initialize model prediction system.")
    raise e


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if data is None or "input" not in data:
            return jsonify({"error": "Missing 'input' key in JSON payload."}), 400

        # List of expected columns in correct order (same as training)
        expected_columns = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course",
        ]

        #  Directly convert to DataFrame
        input_values = data["input"]

        # Ensure it's a list of lists (batch of inputs)
        if isinstance(input_values[0], str):
            input_values = [input_values]  # single record wrapped into batch

        input_df = pd.DataFrame(input_values, columns=expected_columns)

        # Predict using DataFrame
        predictions = predictor.predict(input_df)
        predictor.save_predictions(predictions)

        return jsonify({
            "predictions": predictions.tolist()
        }), 200

    except StudentPerformanceError as spe:
        logger.error("StudentPerformanceError during prediction.", exc_info=True)
        return jsonify({"error": str(spe)}), 500

    except Exception as e:
        logger.exception("Unexpected error in /predict endpoint.")
        return jsonify({"error": str(e)}), 500



@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
