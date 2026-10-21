"""Спільні фікстури: одноразове швидке навчання на синтетичних даних."""
import pytest

from api.app import create_app
from src.config import METADATA_FILENAME, MODEL_FILENAME
from src.data import generate_synthetic_dataset
from src.predictor import PricePredictor
from src.train import run_training

VALID_PAYLOAD = {
    "area": 5000, "bedrooms": 3, "bathrooms": 2, "stories": 2,
    "mainroad": "yes", "guestroom": "no", "basement": "no", "hotwaterheating": "no",
    "airconditioning": "yes", "parking": 1, "prefarea": "no", "furnishingstatus": "semi-furnished",
}


@pytest.fixture(scope="session")
def synthetic_df():
    return generate_synthetic_dataset(n_rows=300, seed=0)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, synthetic_df):
    """Навчає fast-модель один раз на всю сесію й повертає (metadata, models_dir)."""
    root = tmp_path_factory.mktemp("artifacts")
    metadata = run_training(
        synthetic_df, fast=True, models_dir=root / "models", reports_dir=root / "reports", make_plots=False
    )
    return metadata, root / "models"


@pytest.fixture()
def client(trained):
    _, models_dir = trained
    predictor = PricePredictor(models_dir / MODEL_FILENAME, models_dir / METADATA_FILENAME)
    return create_app(predictor).test_client()


@pytest.fixture()
def valid_payload():
    return dict(VALID_PAYLOAD)
