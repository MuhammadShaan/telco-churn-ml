import joblib
import pandas as pd

def predict_new(data_dict):
    df = pd.DataFrame([data_dict])

    model = joblib.load("models/model.pkl")
    threshold = joblib.load("models/threshold.pkl")

    prob = model.predict_proba(df)[0][1]
    prediction = int(prob >= threshold)

    return prob, prediction
