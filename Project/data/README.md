```bash
pip install kaggle            # потрібен токен у ~/.kaggle/kaggle.json
kaggle datasets download -d harishkumardatalab/housing-price-prediction -p data/raw --unzip
```

Очікуваний шлях за замовчуванням: `data/raw/Housing.csv` (цільова колонка `price`).
Якщо файл названий інакше або цільова колонка інша:

```bash
python -m src.train --data data/raw/<файл>.csv --target <колонка_ціни>
```

Для інших датасетів (Vietnam Housing 2024, House Prices – Advanced Regression)
схема визначається автоматично, але похідні ознаки в `src/features.py`
розраховані на колонки `area`, `bedrooms`, `bathrooms` та yes/no-прапорці —
адаптуйте клас `HousingFeatureEngineer` під свій набір.

Без доступу до Kaggle пайплайн можна перевірити на синтетичних даних:
`python -m src.train --synthetic`.
