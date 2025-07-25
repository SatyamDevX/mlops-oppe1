from feast import FeatureStore

# Initialize FeatureStore
store = FeatureStore(repo_path="feature_repo")

# List all features from feature view
feature_view = store.get_feature_view("stock_features")

print("Available features in 'stock_features':")
for feature in feature_view.schema:
    print(f"- {feature.name}: {feature.dtype}")

