from api.app import create_app
from src.predictor import PricePredictor


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is True


def test_predict_returns_ordered_range(client, valid_payload):
    response = client.post("/predict/price", json=valid_payload)
    assert response.status_code == 200
    body = response.get_json()
    assert body["price_range"]["min"] <= body["predicted_price"] <= body["price_range"]["max"]
    assert body["predicted_price"] > 0


def test_predict_is_deterministic_and_case_insensitive(client, valid_payload):
    first = client.post("/predict/price", json=valid_payload).get_json()["predicted_price"]
    valid_payload["mainroad"] = "YES"
    second = client.post("/predict/price", json=valid_payload).get_json()["predicted_price"]
    assert first == second


def test_missing_field_returns_400(client, valid_payload):
    del valid_payload["area"]
    response = client.post("/predict/price", json=valid_payload)
    assert response.status_code == 400
    assert "area" in response.get_json()["details"][0]


def test_invalid_values_return_400(client, valid_payload):
    valid_payload["bedrooms"] = "many"
    valid_payload["furnishingstatus"] = "castle"
    response = client.post("/predict/price", json=valid_payload)
    assert response.status_code == 400
    assert len(response.get_json()["details"]) == 2


def test_out_of_range_value_gives_warning_not_error(client, valid_payload):
    valid_payload["area"] = 10**6
    response = client.post("/predict/price", json=valid_payload)
    assert response.status_code == 200
    assert response.get_json()["warnings"]


def test_non_json_body_returns_415(client):
    assert client.post("/predict/price", data="area=1").status_code == 415


def test_batch_prediction(client, valid_payload):
    response = client.post("/predict/price/batch", json={"instances": [valid_payload, valid_payload]})
    assert response.status_code == 200
    assert response.get_json()["count"] == 2


def test_batch_rejects_empty_instances(client):
    assert client.post("/predict/price/batch", json={"instances": []}).status_code == 400


def test_model_info_exposes_schema(client):
    body = client.get("/model/info").get_json()
    assert "area" in body["schema"]["numeric"]


def test_service_without_model_returns_503(tmp_path, valid_payload):
    predictor = PricePredictor(tmp_path / "missing.joblib", tmp_path / "missing.json")
    client = create_app(predictor).test_client()
    assert client.get("/health").status_code == 503
    assert client.post("/predict/price", json=valid_payload).status_code == 503
