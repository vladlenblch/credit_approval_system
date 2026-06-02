from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.model_service import get_model_service

app = FastAPI()

FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Backend API is running"}


@app.post("/check-features")
async def check_features(payload: dict = Body(...)):
    all_features_received = all(
        feature in payload and payload[feature] not in ("", None)
        for feature in FEATURES
    )
    return {"result": 1 if all_features_received else 0}


@app.post("/predict")
async def predict(payload: dict = Body(...)):
    missing_features = [
        feature
        for feature in FEATURES
        if feature not in payload
    ]

    if missing_features:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required features.",
                "missing_features": missing_features,
            },
        )

    try:
        return get_model_service().predict(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
