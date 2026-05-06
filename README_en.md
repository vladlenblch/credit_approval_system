# Credit Approval ML system

## Installation and local setup

```bash
# clone repository
git clone https://github.com/vladlenblch/credit_approval_system
cd credit_approval_system

# create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# install requirements
pip install -r requirements.txt

# run backend
fastapi dev backend/main.py

# run frontend
python -m http.server 8080 -d frontend

# site will be available at http://localhost:8080/
```

## About

to be completed

### Dataset

[Dataset description](/data/needed_datasets/data_description.md) <br>
[Source dataset](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)

### Hyperparameters

to be completed

### CI/CD

to be completed

## Technology stack and requirements

Backend:
- FastAPI >= 0.136.1

Frontend:
- HTML5
- CSS3
- JavaScript ES6

ML:
- to be completed

## Project structure

- `/backend` - backend API
- `/frontend` - frontend application
- `/ml` - ML layer
- `/data` - project data
