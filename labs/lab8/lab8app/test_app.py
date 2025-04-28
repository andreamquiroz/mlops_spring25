import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "male": 0,
    "age": 70,
    "education": 1,
    "currentSmoker": 1,
    "cigsPerDay": 20,
    "BPMeds": 0,
    "prevalentStroke": 1,
    "prevalentHyp": 0,
    "diabetes": 1,
    "totChol": 218,
    "sysBP": 130,
    "diaBP": 80,
    "BMI": 26.0,
    "heartRate": 75,
    "glucose": 90
}

response = requests.post(url, json=data)
print("Prediction response:", response.json())