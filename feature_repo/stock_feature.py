from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64
from datetime import timedelta

# Entity
stock = Entity(name="stock", join_keys=["stock"], value_type=ValueType.STRING)

# File source (v0)
stock_data_source = FileSource(
    path="data/v0_processed.parquet",
    timestamp_field="timestamp",
    created_timestamp_column=None,
)

# FeatureView
stock_fv = FeatureView(
    name="stock_features",
    entities=[stock],
    ttl=timedelta(days=1),
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Float32),


                # 🆕 Added raw features
        Field(name="open", dtype=Float32),
        Field(name="high", dtype=Float32),
        Field(name="low", dtype=Float32),
        Field(name="close", dtype=Float32),
        Field(name="volume", dtype=Float32),

        Field(name="target", dtype=Int64),
        Field(name="target", dtype=Int64),
    ],
    source=stock_data_source,
)

