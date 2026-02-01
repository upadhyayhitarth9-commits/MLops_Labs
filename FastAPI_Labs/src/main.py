from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from typing import List
import logging
from predict import predict_data, predict_proba, get_feature_importance, get_model_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Wine Quality Classification API",
    description="API for predicting wine class using Random Forest Classifier",
    version="1.0.0"
)

class WineData(BaseModel):
    alcohol: float = Field(..., ge=0)
    malic_acid: float = Field(..., ge=0)
    ash: float = Field(..., ge=0)
    alcalinity_of_ash: float = Field(..., ge=0)
    magnesium: float = Field(..., ge=0)
    total_phenols: float = Field(..., ge=0)
    flavanoids: float = Field(..., ge=0)
    nonflavanoid_phenols: float = Field(..., ge=0)
    proanthocyanins: float = Field(..., ge=0)
    color_intensity: float = Field(..., ge=0)
    hue: float = Field(..., ge=0)
    od280_od315: float = Field(..., ge=0)
    proline: float = Field(..., ge=0)

class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: dict

class BatchPredictionResponse(BaseModel):
    predictions: List[dict]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

@app.get("/", status_code=status.HTTP_200_OK, response_model=HealthResponse)
async def health_check():
    try:
        get_model_info()
        return HealthResponse(status="healthy", model_loaded=True)
    except Exception:
        return HealthResponse(status="healthy", model_loaded=False)

@app.get("/model/info", status_code=status.HTTP_200_OK)
async def model_info():
    try:
        info = get_model_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/features", status_code=status.HTTP_200_OK)
async def feature_importance():
    try:
        importance = get_feature_importance()
        return {"feature_importance": importance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", status_code=status.HTTP_200_OK, response_model=PredictionResponse)
async def predict_wine(wine_data: WineData):
    try:
        features = [[
            wine_data.alcohol, wine_data.malic_acid, wine_data.ash,
            wine_data.alcalinity_of_ash, wine_data.magnesium, wine_data.total_phenols,
            wine_data.flavanoids, wine_data.nonflavanoid_phenols, wine_data.proanthocyanins,
            wine_data.color_intensity, wine_data.hue, wine_data.od280_od315, wine_data.proline
        ]]
        prediction = predict_data(features)[0]
        probabilities = predict_proba(features)[0]
        class_names = ["class_0", "class_1", "class_2"]
        prob_dict = {name: float(prob) for name, prob in zip(class_names, probabilities)}
        return PredictionResponse(
            prediction=int(prediction),
            class_name=class_names[prediction],
            probabilities=prob_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", status_code=status.HTTP_200_OK, response_model=BatchPredictionResponse)
async def predict_batch(wine_samples: List[WineData]):
    try:
        features = [
            [w.alcohol, w.malic_acid, w.ash, w.alcalinity_of_ash, w.magnesium,
             w.total_phenols, w.flavanoids, w.nonflavanoid_phenols, w.proanthocyanins,
             w.color_intensity, w.hue, w.od280_od315, w.proline]
            for w in wine_samples
        ]
        predictions = predict_data(features)
        probabilities = predict_proba(features)
        class_names = ["class_0", "class_1", "class_2"]
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            prob_dict = {name: float(prob) for name, prob in zip(class_names, probs)}
            results.append({
                "sample_index": i,
                "prediction": int(pred),
                "class_name": class_names[pred],
                "probabilities": prob_dict
            })
        return BatchPredictionResponse(predictions=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
