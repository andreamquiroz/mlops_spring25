from fastapi import FastAPI, Request
from pydantic import BaseModel
import mlflow
import pandas as pd

app = FastAPI()

# Load model ONCE at startup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
model = mlflow.sklearn.load_model("models:/log-reg/2")  # << Updated to point to your registered model

class PatientData(BaseModel):
    male: int
    age: float
    education: float
    currentSmoker: int
    cigsPerDay: float
    BPMeds: int
    prevalentStroke: int
    prevalentHyp: int
    diabetes: int
    totChol: float
    sysBP: float
    diaBP: float
    BMI: float
    heartRate: float
    glucose: float

@app.post("/predict")
def predict(data: PatientData):
    df = pd.DataFrame([data.dict()])
    prediction_proba = model.predict_proba(df)[0][1]
    return {"prediction_probability": prediction_proba}