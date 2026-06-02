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

ML service for credit risk scoring based on borrower data. The API returns default probability and credit approval decision using a weighted blend ensemble.

### Dataset

[Dataset description](/data/data_description.md) <br>
[Source dataset](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)

Preprocessing:

- stratified split
- drop `Unnamed: 0`
- fill missing values with train medians
- missing flags and feature engineering

Output:

- `/data/processed/train.csv`
- `/data/processed/valid.csv`
- `/data/processed/test.csv`
- `/data/processed/prepare_data_metadata.json` - preparation parameters

### Models

The final ensemble uses the following models:

- `CatBoostClassifier`
- `LGBMClassifier`
- `XGBClassifier`

Hyperparameter optimization was performed separately for each model using `Optuna`. Model performance was evaluated using stratified cross-validation with `StratifiedKFold` with 3 folds.

Final result:

- OOF predictions for each model
- weighted blend with weights optimized by `Optuna`
- `/predict` API endpoint for inference

## Technology stack

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

## Project structure

- `/backend` - backend API
- `/frontend` - frontend application
- `/ml` - ML layer
- `/data` - original and preprocessed datasets
