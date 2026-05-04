from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()

# Load model and scalers
model = load_model("model.keras")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")


class SensorInput(BaseModel):
    R: float
    G: float
    B: float


@app.get("/")
def home():
    return {"message": "Model is running"}


@app.post("/predict")
def predict(data: SensorInput):

    R = data.R
    G = data.G
    B = data.B

    eps = 1e-9
    total = R + G + B + eps

    R_norm = R / total
    G_norm = G / total
    B_norm = B / total

    RG = R / (G + eps)
    RB = R / (B + eps)
    GB = G / (B + eps)

    features = np.array([[
        R, G, B,
        R_norm, G_norm, B_norm,
        RG, RB, GB
    ]])

    X_scaled = scaler_X.transform(features)

    pred_scaled = model.predict(X_scaled)
    pred = scaler_y.inverse_transform(pred_scaled)

    theta = pred[0][0]
    phi_sin = pred[0][1]
    phi_cos = pred[0][2]
    distance = pred[0][3]

    phi = np.degrees(np.arctan2(phi_sin, phi_cos))
    phi = (phi + 360) % 360

    return {
        "theta": float(theta),
        "phi": float(phi),
        "distance": float(distance)
    }