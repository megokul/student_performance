# 🎓 StudentPerformance: Modular ML Pipeline for Academic Score Prediction

> 🚀 A fully modular, production-grade ML pipeline for student performance prediction. Built using FastAPI, DVC, Optuna, MLflow, PostgreSQL, and Docker — following industry-grade MLOps principles.

---

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow)
![DVC](https://img.shields.io/badge/DVC-Data_Versioning-purple?logo=dvc)
![Optuna](https://img.shields.io/badge/Optuna-HPO-orange?logo=optuna)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791?logo=postgresql)
![AWS](https://img.shields.io/badge/AWS-S3-yellow?logo=amazonaws)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Modeling-F7931E?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Dataframe-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Array-013243?logo=numpy)

---

## ✅ Features

* ✅ End-to-end ML pipeline: Ingestion ➜ Validation ➜ Transformation ➜ Training ➜ Evaluation ➜ Prediction
* ✅ YAML-driven configuration system with full ConfigBox support
* ✅ Dynamic preprocessing (scaling, encoding, imputation, custom ops)
* ✅ Optuna hyperparameter tuning with MLflow tracking
* ✅ S3 + local storage support for all artifacts and logs
* ✅ PostgreSQL-backed ingestion with dynamic table creation from schema
* ✅ DVC-integrated dataset management

---

## 📂 Project Structure

```text
student_performance/
├── app.py                   # FastAPI app for /predict and /train
├── Dockerfile               # Container build setup
├── docker-compose.yaml      # FastAPI + Redis + Celery stack
├── config/                  # All YAML configs: params, schema, templates
├── data/                    # DVC-tracked raw/validated/transformed data
├── artifacts/               # Timestamped pipeline outputs
├── logs/                    # UTC-based log directory
├── templates/               # HTML Jinja templates (for UI if enabled)
├── requirements.txt         # Dependency list
└── src/student_performance/
    ├── components/          # Pipeline stages
    ├── config/              # Configuration manager
    ├── constants/           # Path constants and global flags
    ├── data_processors/     # Reusable preprocessing modules
    ├── dbhandler/           # PostgreSQL and S3 handler classes
    ├── entity/              # Dataclass config and artifact entities
    ├── exception/           # Central error handling
    ├── inference/           # Inference model wrapper
    ├── logging/             # Centralized log setup (local/S3)
    ├── pipeline/            # Training and prediction pipeline runners
    ├── utils/               # File I/O and transformation helpers
    └── worker/              # Celery task runner
```

---

## 🔁 Pipeline Flow

```text
PostgreSQL → Ingestion → Validation → Transformation → Training → Evaluation → Inference Model
```

---

## ⚙️ Configuration

The system uses declarative YAML for all configuration and parameter tuning.

**Config Files:**
- `config.yaml`: Paths, filenames, directories
- `params.yaml`: ML params, splits, methods, tuning spaces
- `schema.yaml`: Data schema + validation constraints
- `templates.yaml`: Structure of report templates

**Secrets (.env):**
```dotenv
# PostgreSQL
PG_USER=
PG_PASSWORD=
PG_HOST=
PG_PORT=5432
PG_DB=student_performance_db

# AWS
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=

# MLflow
MLFLOW_TRACKING_URI=
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
```

---

## 🧪 Run Instructions

### ⚙️ Run FastAPI app locally

```bash
uvicorn app:app --reload
```

### 🐳 Run with Docker Compose

```bash
docker compose up --build
```

---

## 🔬 MLflow Tracking

* Experiment: `StudentPerformanceExperiment`
* Registry: `StudentPerformanceModel`
* Metrics: `neg_root_mean_squared_error`, `r2`, `mae`, `adjusted_r2`, etc.

```bash
mlflow ui
```

---

## 📊 FastAPI Endpoints

- `POST /predict` → Accepts input array or CSV for inference
- `POST /train` → Triggers background model training via Celery

---

## 📜 License

This project is licensed under **GPLv3**.

---

## 👨‍💻 Author

**Gokul Krishna N V**  
Machine Learning Engineer — UK 🇬🇧  
[GitHub](https://github.com/megokul) • [LinkedIn](https://www.linkedin.com/in/nv-gokul-krishna)
