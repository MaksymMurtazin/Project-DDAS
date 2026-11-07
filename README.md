# Прогнозування цін нерухомості — end-to-end сервіс

Реалізація **Проєкту 1 "Прогнозування цiн нерухомостi"**:
навчання моделей (Random Forest, XGBoost, LightGBM + ансамбль) → REST API (Flask + gunicorn) →
Streamlit-інтерфейс → Docker.

## Структура

```
house-price-service/
├── src/
│   ├── config.py       # шляхи, константи, логування
│   ├── data.py         # завантаження/очищення CSV, синтетичні дані
│   ├── features.py     # інженерія ознак (sklearn-трансформер) + схема входу
│   ├── modeling.py     # Pipeline, кандидати, CV, тюнінг, інтервал прогнозу
│   ├── train.py        # CLI навчання, збереження артефактів
│   ├── predictor.py    # завантаження моделі, валідація входу, прогноз
│   └── reporting.py    # графіки
├── api/app.py          # Flask REST API
├── ui/streamlit_app.py # дашборд
├── docker/             # Dockerfile.api, Dockerfile.ui
├── docker-compose.yml  # api + ui (+ trainer за профілем)
├── tests/              # pytest (20 тестів)
├── data/README.md      # як отримати датасет
├── models/  reports/   # артефакти навчання (створюються train.py)
└── requirements*.txt, Makefile
```

## Швидкий старт (локально)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -r requirements-ui.txt

# 1. Дані: див. data/README.md  (або --synthetic для перевірки пайплайна)
python -m src.train --data data/raw/Housing.csv
python -m src.train --tune --n-iter 30          # з підбором гіперпараметрів

# 2. API
python -m api.app                                 # dev, http://127.0.0.1:5000
gunicorn "api.app:create_app()"                   # prod-подібний запуск

# 3. UI
API_URL=http://localhost:5000 streamlit run ui/streamlit_app.py

# 4. Тести
pytest -q
```

## Docker

```bash
docker compose --profile train run --rm trainer   # навчання (потрібен data/raw/Housing.csv)
docker compose up --build                         # API :5000, UI :8501
```

## API

```bash
curl -s localhost:5000/predict/price -H 'Content-Type: application/json' -d '{
  "area": 5000, "bedrooms": 3, "bathrooms": 2, "stories": 2, "mainroad": "yes",
  "guestroom": "no", "basement": "no", "hotwaterheating": "no", "airconditioning": "yes",
  "parking": 1, "prefarea": "no", "furnishingstatus": "semi-furnished"}'
```

```json
{"predicted_price": 8051258.94,
 "price_range": {"min": 6485603.78, "max": 10229614.65, "confidence_level": 0.9},
 "model_version": "20261012-092759", "warnings": []}
```

| Метод | Шлях | Опис |
|---|---|---|
| GET | `/health` | 200 якщо модель завантажена, інакше 503 |
| GET | `/model/info` | схема входу, метрики, порівняння моделей |
| POST | `/predict/price` | один об'єкт |
| POST | `/predict/price/batch` | `{"instances": [...]}`, до 1000 об'єктів |

Коди відповідей: `400` — помилки валідації (повертається список усіх проблем одразу),
`415` — не JSON, `503` — модель не натренована, `500` — внутрішня помилка (без деталей у відповіді, traceback у логах).


## Про метрики

* Ціль — точність 5–7 % від ринкової вартості. Скрипт чесно звітує частку прогнозів у межах ±5 % та ±7 % (`within_5pct`, `within_7pct`) і MAPE.
* Невеликий датасет `Housing.csv` (~545 рядків) може не досягти цієї цілі — це властивість даних, а не помилка пайплайна.
* Метрики на `--synthetic` даних не мають змісту для висновків: синтетика потрібна лише для перевірки роботи коду.
* Якщо запускали `--tune`, довіряйте метрикам на hold-out тесті, а не CV (CV після підбору параметрів оптимістична).
