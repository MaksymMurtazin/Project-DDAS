import numpy as np
import pandas as pd
import pytest

from src.data import DataError, clean_dataset, load_dataset
from src.features import HousingFeatureEngineer, build_schema


def test_load_missing_file_raises_data_error(tmp_path):
    with pytest.raises(DataError):
        load_dataset(tmp_path / "nope.csv", "price")


def test_clean_dataset_requires_target(synthetic_df):
    with pytest.raises(DataError):
        clean_dataset(synthetic_df.drop(columns=["price"]), "price")


def test_clean_dataset_drops_invalid_target_and_duplicates(synthetic_df):
    dirty = pd.concat([synthetic_df, synthetic_df.head(5)], ignore_index=True)
    dirty.loc[100, "price"] = -1  # рядки поза дублікатами head(5)
    dirty.loc[101, "price"] = np.nan
    cleaned = clean_dataset(dirty, "price")
    assert len(cleaned) == len(synthetic_df) - 2
    assert (cleaned["price"] > 0).all()


def test_feature_engineer_adds_features_and_maps_binary(synthetic_df):
    X = synthetic_df.drop(columns=["price"])
    out = HousingFeatureEngineer().fit(X).transform(X)
    assert {"total_rooms", "area_per_room", "amenity_count"} <= set(out.columns)
    assert set(out["mainroad"].unique()) <= {0.0, 1.0}
    assert out["furnishingstatus"].dtype == object  # багатокласова категорія лишається текстом


def test_feature_engineer_single_row_is_case_insensitive(synthetic_df):
    X = synthetic_df.drop(columns=["price"])
    engineer = HousingFeatureEngineer().fit(X)
    row = X.head(1).copy()
    row["mainroad"] = " YES "
    assert engineer.transform(row)["mainroad"].iloc[0] == 1.0


def test_schema_describes_columns(synthetic_df):
    schema = build_schema(synthetic_df.drop(columns=["price"]))
    assert "area" in schema["numeric"] and schema["numeric"]["area"]["is_integer"]
    assert schema["categorical"]["mainroad"] == ["no", "yes"]
