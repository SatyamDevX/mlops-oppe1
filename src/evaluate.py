# src/evaluate.py

import pandas as pd
import joblib
import os
from feast import FeatureStore
from sklearn.metrics import accuracy_score, f1_score

# ---- Paths setup ----
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURE_REPO_PATH = os.path.join(ROOT_DIR, "feature_repo")
DATA_PATH = os.path.join(FEATURE_REPO_PATH, "data", "v0_processed.parquet")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "random_forest.pkl")
REPORT_PATH = os.path.join(ROOT_DIR, "reports", "eval_report.txt")

def fetch_data_from_feast():
    store = FeatureStore(repo_path=FEATURE_REPO_PATH)

    features = [
        "stock_features:rolling_avg_10",
        "stock_features:volume_sum_10",
        "stock_features:volume",
        "stock_features:open",
        "stock_features:high",
        "stock_features:low",
        "stock_features:close",
        "stock_features:target"
    ]

    entity_df = pd.read_parquet(DATA_PATH)

    df = store.get_historical_features(
        entity_df=entity_df,
        features=features
    ).to_df()

    return df

def main():
    print("🔍 Loading model and data for evaluation...")
    df = fetch_data_from_feast()
    df.dropna(inplace=True)

    X = df.drop(columns=["target", "event_timestamp", "stock"])
    y = df["target"]

    # Load trained model from .pkl
    model = joblib.load(MODEL_PATH)

    # Predict and evaluate
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    with open(REPORT_PATH, "w") as f:
        f.write("## 📊 Model Evaluation Metrics\n")
        f.write(f"**Accuracy**: {acc:.4f}\n")
        f.write(f"**F1 Score**: {f1:.4f}\n")

    print("✅ Evaluation complete")
    print(f"📄 Report saved to {REPORT_PATH}")

if __name__ == "__main__":
    main()

