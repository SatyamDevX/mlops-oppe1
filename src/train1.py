# train.py

import pandas as pd
from feast import FeatureStore
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
import os

def fetch_data_from_feast():
    print("Connecting to FeatureStore...")
    store = FeatureStore(repo_path="../feature_repo")

    # Load a small portion of the v0_processed dataset
    df = pd.read_parquet("../feature_repo/data/v0_processed.parquet")

    # Select only the columns needed for entity_df
    entity_df = df[["stock", "timestamp"]].drop_duplicates().reset_index(drop=True)

    # Avoid memory issues: limit rows
    entity_df = entity_df.head(5000)

    # Feast expects 'event_timestamp' explicitly
    entity_df = entity_df.rename(columns={"timestamp": "event_timestamp"})

    print("Fetching features from Feast...")
    training_df = store.get_historical_features(
        entity_df=entity_df.tail(100),
        features=[
            "stock_features:rolling_avg_10",
            "stock_features:volume_sum_10",
            "stock_features:target"
        ],
    ).to_df()

    print("Successfully fetched training data.")
    return training_df

def main():
    df = fetch_data_from_feast()

    print("Preprocessing data...")
    df = df.dropna()
    X = df[["rolling_avg_10", "volume_sum_10"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Evaluation Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/random_forest.pkl")
    print("Model saved at ../models/random_forest.pkl")

if __name__ == "__main__":
    main()

