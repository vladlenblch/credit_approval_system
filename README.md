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

[Описание датасета](/data/needed_datasets/data_description.md) <br>
[Исходный датасет](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)

### Гиперпараметры

to be completed

### CI/CD

to be completed

## Технологический стек и требования

Backend:
- FastAPI >= 0.136.1

Frontend:
- HTML5
- CSS3
- JavaScript ES6

ML:
- to be completed

## Структура проекта

- `/backend` - бэкенд-часть проекта с API
- `/frontend` - фронтенд-часть проекта
- `/ml` - ML-слой проекта
- `/data` - данные проекта
