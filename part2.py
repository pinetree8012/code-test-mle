import json
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
import time


MODEL_PATH: str = "fraud_prevention_model.pt"
MEAN: list[float] = [-2.30926389e-17, -9.94759830e-17, -7.46069873e-17, 1.42108547e-17]
STD: list[float] = [1.0, 1.0, 1.0, 1.0]

app = FastAPI(title="Fraud Detection Inference API")

device = torch.device("cpu")
model = None


# Schemas
class Transaction(BaseModel):
    amount: float = Field(...)
    time: int = Field(...)
    mismatch: int = Field(...)
    frequency: int = Field(...)


class PredictResponse(BaseModel):
    fraud_probability: float = Field(...)
    is_fraud: bool = Field(...)


@app.on_event("startup")
def on_startup() -> None:
    """Load the model on startup."""
    global model
    try:   
        start = time.time()
        model = torch.jit.load(MODEL_PATH, map_location=device)
        model.eval()
        end = time.time()
        model_load_time = end - start
        print("Model Load Time: ", model_load_time)
    except FileNotFoundError:
        model = None


@app.post("/predict", response_model=PredictResponse)
async def predict_fraud(transaction: Transaction) -> PredictResponse:
    """Predict fraud for a single transaction."""
    if model is None:
        return {"error": "Model not loaded."}

    # Build feature vector
    features = np.array([[
        transaction.amount,
        transaction.time,
        transaction.mismatch,
        transaction.frequency,
    ]], dtype=np.float32)

    # Normalise using training-time mean and std
    features_normalized = (features - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    input_tensor = torch.from_numpy(features_normalized)

    # Inference
    with torch.inference_mode():
        logit = model(input_tensor)
        # Convert logits to probability via sigmoid
        probability = torch.sigmoid(logit).item()

    # Threshold
    is_fraud = probability > 0.5

    return PredictResponse(
        fraud_probability=round(probability, 4),
        is_fraud=bool(is_fraud),
    )


@app.get("/healthz")
def healthz() -> dict:
    ok = model is not None
    return {"status": "ok" if ok else "not_ready"}
