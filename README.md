# Credit Approval ML system

## Установка и запуск локально

```bash
# клонировать репозиторий
git clone https://github.com/vladlenblch/credit_approval_system
cd credit_approval_system

# создать и активировать виртуальное окружение
python -m venv .venv
source .venv/bin/activate

# установить зависимости
pip install -r requirements.txt

# запустить бэкенд
fastapi dev backend/main.py

# запустить фронтенд
python -m http.server 8080 -d frontend

# сайт будет доступен по адресу http://localhost:8080/
```

## О проекте

to be completed

### Датасет

[Описание датасета](/data/data_description.md) <br>
[Исходный датасет](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)

Препроцессинг:

- сплит с сохранением пропорций
- удаление `Unnamed: 0`
- заполнение пропусков медианами train
- missing-флаги и новые фичи

Результат:

- `/data/processed/train.csv`
- `/data/processed/valid.csv`
- `/data/processed/test.csv`
- `/data/processed/prepare_data_metadata.json` - параметры подготовки

### Модели

Для стекинга были использованы следующие модели:

- `CatBoostClassifier`
- `LGBMClassifier`
- `XGBClassifier`

Для каждой модели был выполнен подбор гиперпараметров с использованием `Optuna`. Оценка качества на каждой итерации оптимизации проводилась с помощью стратифицированной кросс-валидации `StratifiedKFold` с 3 фолдами.

## Технологический стек

Backend:

- FastAPI

Frontend:

- HTML5
- CSS3
- JavaScript ES6

ML:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Optuna
- CatBoost
- LightGBM
- XGBoost

## Структура проекта

- `/backend` - бэкенд-часть проекта с API
- `/frontend` - фронтенд-часть проекта
- `/ml` - ML-слой проекта
- `/data` - исходные и предобработанные датасеты
