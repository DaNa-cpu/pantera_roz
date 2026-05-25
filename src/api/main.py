from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="Payment Delay Prediction API",
    version="1.0.0",
    description="FastAPI service for real-time inference on telecom payment delay risk.",
)

TARGET_COLUMN = "payment_delay"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

FEATURE_COLUMNS = [
    "state",
    "account_length",
    "area_code",
    "international_plan",
    "voice_mail_plan",
    "number_vmail_messages",
    "total_day_minutes",
    "total_day_calls",
    "total_day_charge",
    "total_eve_minutes",
    "total_eve_calls",
    "total_eve_charge",
    "total_night_minutes",
    "total_night_calls",
    "total_night_charge",
    "total_intl_minutes",
    "total_intl_calls",
    "total_intl_charge",
    "number_customer_service_calls",
]

NUMERIC_COLUMNS = [
    "account_length",
    "number_vmail_messages",
    "total_day_minutes",
    "total_day_calls",
    "total_day_charge",
    "total_eve_minutes",
    "total_eve_calls",
    "total_eve_charge",
    "total_night_minutes",
    "total_night_calls",
    "total_night_charge",
    "total_intl_minutes",
    "total_intl_calls",
    "total_intl_charge",
    "number_customer_service_calls",
]

CLASS_LABELS = {
    0: "no",
    1: "yes",
}


class TelecomCustomerFeatures(BaseModel):
    state: str = Field(..., examples=["HI"])
    account_length: int = Field(..., ge=0, examples=[33])
    area_code: str = Field(..., examples=["area_code_415"])
    international_plan: Literal["yes", "no"] = Field(..., examples=["no"])
    voice_mail_plan: Literal["yes", "no"] = Field(..., examples=["no"])
    number_vmail_messages: int = Field(..., ge=0, examples=[0])
    total_day_minutes: float = Field(..., ge=0, examples=[200.5])
    total_day_calls: int = Field(..., ge=0, examples=[117])
    total_day_charge: float = Field(..., ge=0, examples=[34.09])
    total_eve_minutes: float = Field(..., ge=0, examples=[159.9])
    total_eve_calls: int = Field(..., ge=0, examples=[111])
    total_eve_charge: float = Field(..., ge=0, examples=[13.59])
    total_night_minutes: float = Field(..., ge=0, examples=[196.2])
    total_night_calls: int = Field(..., ge=0, examples=[84])
    total_night_charge: float = Field(..., ge=0, examples=[8.83])
    total_intl_minutes: float = Field(..., ge=0, examples=[16.3])
    total_intl_calls: int = Field(..., ge=0, examples=[6])
    total_intl_charge: float = Field(..., ge=0, examples=[4.4])
    number_customer_service_calls: int = Field(..., ge=0, examples=[3])


class PredictionResponse(BaseModel):
    prediction: Literal["yes", "no"]
    target: str
    probabilities: dict[str, float] | None = None
    model_status: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    feature_count: int
    model_status: str


class ModelArtifacts:
    def __init__(self) -> None:
        self.model = joblib.load(MODEL_PATH)
        self.preprocessor = joblib.load(PREPROCESSOR_PATH)
        self.scaler = joblib.load(SCALER_PATH)


@lru_cache(maxsize=1)
def get_artifacts() -> ModelArtifacts:
    return ModelArtifacts()


def format_feature_name(name: str) -> str:
    return name.replace("cat__", "").replace("remainder__", "")


def prepare_model_input(
    features: dict[str, object],
    artifacts: ModelArtifacts,
) -> pd.DataFrame:
    raw_input = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    transformed = artifacts.preprocessor.transform(raw_input)
    transformed_columns = [
        format_feature_name(column)
        for column in artifacts.preprocessor.get_feature_names_out()
    ]
    model_input = pd.DataFrame(transformed, columns=transformed_columns)
    model_input[NUMERIC_COLUMNS] = artifacts.scaler.transform(model_input[NUMERIC_COLUMNS])

    expected_columns = list(artifacts.model.feature_names_in_)
    return model_input[expected_columns]


def predict_with_model(features: dict[str, object]) -> dict[str, object]:
    artifacts = get_artifacts()
    model_input = prepare_model_input(features, artifacts)

    raw_prediction = artifacts.model.predict(model_input)[0]
    prediction = CLASS_LABELS.get(int(raw_prediction), str(raw_prediction))
    response: dict[str, object] = {
        "prediction": prediction,
        "target": TARGET_COLUMN,
        "model_status": "loaded",
    }

    if hasattr(artifacts.model, "predict_proba"):
        probabilities = artifacts.model.predict_proba(model_input)[0]
        response["probabilities"] = {
            CLASS_LABELS.get(int(class_name), str(class_name)): float(probability)
            for class_name, probability in zip(artifacts.model.classes_, probabilities, strict=True)
        }

    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_artifacts()
    except Exception:
        return HealthResponse(
            status="ok",
            model_loaded=False,
            feature_count=len(FEATURE_COLUMNS),
            model_status="model_artifacts_not_available",
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
        feature_count=len(FEATURE_COLUMNS),
        model_status="loaded",
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(features: TelecomCustomerFeatures) -> PredictionResponse:
    payload = (
        features.model_dump()
        if hasattr(features, "model_dump")
        else features.dict()
    )
    try:
        prediction = predict_with_model(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model inference failed: {exc}",
        ) from exc

    return PredictionResponse(**prediction)


@app.get("/features")
def features() -> dict[str, list[str]]:
    return {"features": FEATURE_COLUMNS}
