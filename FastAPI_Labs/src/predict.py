import joblib
import json
import numpy as np

model = None
scaler = None
metadata = None

def load_model_artifacts():
    global model, scaler, metadata
    if model is None:
        model = joblib.load("../model/wine_model.pkl")
        scaler = joblib.load("../model/scaler.pkl")
        with open("../model/metadata.json", "r") as f:
            metadata = json.load(f)
    return model, scaler, metadata

def predict_data(X):
    model, scaler, _ = load_model_artifacts()
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    return y_pred

def predict_proba(X):
    model, scaler, _ = load_model_artifacts()
    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)
    return probabilities

def get_feature_importance():
    model, _, metadata = load_model_artifacts()
    importance = model.feature_importances_
    feature_names = metadata["feature_names"]
    importance_dict = {
        name: float(score) 
        for name, score in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    }
    return importance_dict

def get_model_info():
    _, _, metadata = load_model_artifacts()
    return metadata
