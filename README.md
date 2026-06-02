# Credit Approval ML system

## О проекте

ML-сервис для оценки кредитного риска по данным заемщика. API возвращает вероятность дефолта и решение о выдаче кредита на основе weighted blend ансамбля.

## Примеры

![Approve example](assets/example_approve.png)

![Reject example](assets/example_reject.png)

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

В финальном ансамбле используются следующие модели:

- `CatBoostClassifier`
- `LGBMClassifier`
- `XGBClassifier`

Для каждой модели был выполнен подбор гиперпараметров с использованием `Optuna`. Оценка качества проводилась с помощью стратифицированной кросс-валидации `StratifiedKFold` с 3 фолдами.

Финальный результат:

- OOF-предсказания для каждой модели
- weighted blend по весам, подобранным через `Optuna`
- `/predict` API endpoint для инференса

Метрики weighted blend:

- OOF PR-AUC: `0.4058`
- Test PR-AUC: `0.4095`

### Бизнес-решение

Порог отказа подбирается отдельно на OOF-предсказаниях по простой profit-функции:

- хороший одобренный клиент: `+1`
- дефолтный одобренный клиент: `-5`

Итоговый threshold: `0.381`.

На test:

- profit: `16234`
- approval rate: `0.8975`
- bad recall rejected: `0.5612`

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
- `/data` - исходные и предобработанные датасеты
- `/ml` - ML-слой проекта
- `/artifacts` - локальные артефакты моделей и ансамбля
- `README.md` - описание проекта
- `README_en.md` - описание проекта на английском
