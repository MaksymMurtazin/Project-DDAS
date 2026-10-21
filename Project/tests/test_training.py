import json

from src.config import METADATA_FILENAME, MODEL_FILENAME


def test_artifacts_are_saved(trained):
    metadata, models_dir = trained
    assert (models_dir / MODEL_FILENAME).is_file()
    saved = json.loads((models_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert saved["model_version"] == metadata["model_version"]


def test_model_beats_baseline(trained):
    metadata, _ = trained
    comparison = metadata["model_comparison"]
    assert comparison["ensemble"]["test"]["rmse"] < comparison["baseline_median"]["test"]["rmse"]
    assert metadata["metrics"]["test"]["r2"] > 0.3


def test_interval_contains_point_estimate(trained):
    metadata, _ = trained
    interval = metadata["prediction_interval"]
    assert 0 < interval["lower_ratio"] <= 1.0 <= interval["upper_ratio"]
