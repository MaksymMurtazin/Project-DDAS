PY ?= python

.PHONY: install train train-synthetic serve test ui up train-docker

install:            ## встановити залежності (розробка)
	$(PY) -m pip install -r requirements-dev.txt -r requirements-ui.txt

train:              ## навчити модель на data/raw/Housing.csv
	$(PY) -m src.train

train-synthetic:    ## навчити на синтетичних даних (перевірка пайплайна)
	$(PY) -m src.train --synthetic

serve:              ## dev-сервер API на :5000
	$(PY) -m api.app

ui:                 ## Streamlit-інтерфейс на :8501
	API_URL=http://localhost:5000 streamlit run ui/streamlit_app.py

test:               ## юніт-тести
	$(PY) -m pytest -q

train-docker:       ## навчання в контейнері
	docker compose --profile train run --rm trainer

up:                 ## API + UI у Docker
	docker compose up --build
