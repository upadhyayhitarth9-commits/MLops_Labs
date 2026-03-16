from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from typing import List
import logging
from predict import predict_data, predict_proba, get_feature_importance, get_model_info

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Wine Quality Classification API",
    description="API for predicting wine class using Random Forest Classifier",
    version="1.0.0"
)

# Request/Response Models
class WineData(BaseModel):
    alcohol: float = Field(..., description="Alcohol content", ge=0)
    malic_acid: float = Field(..., description="Malic acid content", ge=0)
    ash: float = Field(..., description="Ash content", ge=0)
    alcalinity_of_ash: float = Field(..., description="Alcalinity of ash", ge=0)
    magnesium: float = Field(..., description="Magnesium content", ge=0)
    total_phenols: float = Field(..., description="Total phenols", ge=0)
    flavanoids: float = Field(..., description="Flavanoids content", ge=0)
    nonflavanoid_phenols: float = Field(..., description="Nonflavanoid phenols", ge=0)
    proanthocyanins: float = Field(..., description="Proanthocyanins content", ge=0)
    color_intensity: float = Field(..., description="Color intensity", ge=0)
    hue: float = Field(..., description="Hue value", ge=0)
    od280_od315: float = Field(..., description="OD280/OD315 of diluted wines", ge=0)
    proline: float = Field(..., description="Proline content", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "alcohol": 13.0,
                "malic_acid": 2.0,
                "ash": 2.5,
                "alcalinity_of_ash": 19.0,
                "magnesium": 100.0,
                "total_phenols": 2.5,
                "flavanoids": 2.5,
                "nonflavanoid_phenols": 0.3,
                "proanthocyanins": 1.5,
                "color_intensity": 5.0,
                "hue": 1.0,
                "od280_od315": 3.0,
                "proline": 1000.0
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: dict

class BatchPredictionResponse(BaseModel):
    predictions: List[dict]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

# Endpoints
@app.get("/", status_code=status.HTTP_200_OK, response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        get_model_info()
        return HealthResponse(status="healthy", model_loaded=True)
    except Exception:
        return HealthResponse(status="healthy", model_loaded=False)

@app.get("/model/info", status_code=status.HTTP_200_OK)
async def model_info():
    """Get model metadata and information."""
    try:
        info = get_model_info()
        logger.info("Model info retrieved successfully")
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/features", status_code=status.HTTP_200_OK)
async def feature_importance():
    """Get feature importance scores."""
    try:
        importance = get_feature_importance()
        logger.info("Feature importance retrieved successfully")
        return {"feature_importance": importance}
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", status_code=status.HTTP_200_OK, response_model=PredictionResponse)
async def predict_wine(wine_data: WineData):
    """Predict wine class for a single sample."""
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
        
        logger.info(f"Prediction made: {prediction}")
        return PredictionResponse(
            prediction=int(prediction),
            class_name=class_names[prediction],
            probabilities=prob_dict
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", status_code=status.HTTP_200_OK, response_model=BatchPredictionResponse)
async def predict_batch(wine_samples: List[WineData]):
    """Predict wine class for multiple samples."""
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
        
        logger.info(f"Batch prediction made for {len(wine_samples)} samples")
        return BatchPredictionResponse(predictions=results)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))