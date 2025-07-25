# 📈 Minute-Level Stock Movement Prediction Pipeline

This project predicts minute-level stock price movements on NSE data using a machine learning pipeline that follows modern **MLOps** practices. It uses the past 10 minutes of technical indicators to predict stock price movement 5 minutes ahead.

---

## 🧠 Objective

Predict whether a stock will go **up** or **down** 5 minutes into the future using:
- Historical minute-level data (2018–2021)
- Technical analysis features (moving averages, RSI, etc.)

---

## 🛠️ Tools & Technologies

| Category          | Tools Used                                  |
|------------------|----------------------------------------------|
| ML Models         | XGBoost, Random Forest, etc.                |
| MLOps Versioning  | DVC for data and model versioning           |
| Experimentation   | MLflow for hyperparameter tuning & tracking |
| Feature Store     | Feast                                       |
| CI/CD             | GitHub Actions with CML                     |
| Orchestration     | Shell scripts, Python                       |
| Cloud Platform    | Google Cloud Platform (GCS, Vertex AI)      |

---

## 📁 Project Structure
Stock_prediction_pipeline/
├── data/ # Raw and processed data
│ └── stock_minute_data.csv
├── dvc.yaml # DVC pipeline configuration
├── feature_repo/ # Feast feature store
│ ├── feature_store.yaml
│ ├── example_feature_view.py
│ └── driver.py
├── models/ # Stored trained models
├── src/ # Source code
│ ├── train.py
│ ├── evaluate.py
│ └── preprocess.py
├── evaluate.sh # Runs training, evaluation, and CML report
├── cml.yaml # GitHub Actions workflow for CML
├── params.yaml # Model & training hyperparameters
├── mlruns/ # MLflow tracking
└── README.md # This file



---

## 🔁 Pipeline Overview

### ✅ Step 1: Data Versioning with DVC

- Tracks raw and processed datasets
- Stored remotely on GCS

### ✅ Step 2: Feature Store Integration (Feast)

- Historical features stored & served via Feast
- `feature_repo/` contains views and registry

### ✅ Step 3: Model Training & Evaluation

```bash
python src/train.py
python src/evaluate.py

git clone https://github.com/your-username/Stock_prediction_pipeline.git
cd Stock_prediction_pipeline

# Create virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize DVC and Feast
dvc pull
feast init feature_repo/
feast apply


