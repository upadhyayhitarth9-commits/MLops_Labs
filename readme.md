# Wine Classification API with FastAPI

In this Lab, we will learn how to expose ML models as APIs using FastAPI and uvicorn.

- **FastAPI**: FastAPI is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints.
- **uvicorn**: Uvicorn is an Asynchronous Server Gateway Interface (ASGI) web server implementation for Python. It is often used to serve FastAPI applications.

## Workflow

The workflow involves the following steps:

1. Training a **Random Forest Classifier** on the **Wine Dataset**.
2. Serving the trained model as an API using FastAPI and uvicorn.
3. Testing the API endpoints for predictions.

## Modifications from Original Lab

| Original (Iris) | Modified (Wine) |
|-----------------|-----------------|
| Iris flower dataset (4 features) | Wine dataset (13 features) |
| Decision Tree Classifier | Random Forest Classifier |
| 2 endpoints | 5 endpoints |
| No feature scaling | StandardScaler preprocessing |
| No prediction probabilities | Returns prediction probabilities |
| No feature importance | Feature importance endpoint |

## Setting up the Lab

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the environment:
```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
FastAPI_Labs
├── assets/
├── model/
│   ├── wine_model.pkl
│   ├── scaler.pkl
│   └── metadata.json
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── main.py
│   ├── predict.py
│   └── train.py
├── README.md
└── requirements.txt
```

### File Descriptions

| File | Description |
|------|-------------|
| `data.py` | Loads Wine dataset, splits data, and applies scaling |
| `train.py` | Trains Random Forest model and saves artifacts |
| `predict.py` | Loads model and provides prediction functions |
| `main.py` | FastAPI application with all endpoints |

## Running the Lab

### Step 1: Train the Model

Navigate to the `src/` folder:
```bash
cd src
```

Train the Random Forest Classifier:
```bash
python train.py
```

Expected output:
```
Training Random Forest Classifier...

Model Accuracy: 1.0000

Classification Report:
              precision    recall  f1-score   support

     class_0       1.00      1.00      1.00        15
     class_1       1.00      1.00      1.00        18
     class_2       1.00      1.00      1.00        12

    accuracy                           1.00        45
   macro avg       1.00      1.00      1.00        45
weighted avg       1.00      1.00      1.00        45

Model, scaler, and metadata saved successfully!
```

### Step 2: Start the API Server

```bash
uvicorn main:app --reload
```

### Step 3: Test the Endpoints

Open your browser and navigate to:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check - verifies API and model status |
| `/model/info` | GET | Returns model metadata and accuracy |
| `/model/features` | GET | Returns feature importance scores |
| `/predict` | POST | Predicts wine class for a single sample |
| `/predict/batch` | POST | Predicts wine class for multiple samples |

## Wine Dataset Features

The Wine dataset contains 13 chemical features:

| Feature | Description |
|---------|-------------|
| alcohol | Alcohol percentage |
| malic_acid | Malic acid content |
| ash | Ash content |
| alcalinity_of_ash | Alcalinity of ash |
| magnesium | Magnesium content |
| total_phenols | Total phenolic compounds |
| flavanoids | Flavanoid content |
| nonflavanoid_phenols | Non-flavanoid phenols |
| proanthocyanins | Proanthocyanin content |
| color_intensity | Color intensity |
| hue | Hue value |
| od280_od315 | OD280/OD315 of diluted wines |
| proline | Proline amino acid content |

## Data Models in FastAPI

### 1. WineData Class (Request Model)

```python
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
```

The `WineData` class is a Pydantic model which defines the expected structure of the request body. Features include:
- **Type validation**: All fields must be floats
- **Constraint validation**: All values must be >= 0 (using `ge=0`)
- **Automatic documentation**: Generates OpenAPI schema

### 2. PredictionResponse Class (Response Model)

```python
class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: dict
```

The `PredictionResponse` class defines the structure of the prediction response:
- `prediction`: The predicted class (0, 1, or 2)
- `class_name`: Human-readable class name
- `probabilities`: Confidence scores for each class

## Testing the API

### Using Swagger UI

1. Go to http://127.0.0.1:8000/docs
2. Click on **POST /predict**
3. Click **"Try it out"**
4. Enter the wine features in JSON format
5. Click **"Execute"**

### Using curl

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Using Python

```python
import requests

url = "http://127.0.0.1:8000/predict"
wine_sample = {
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

response = requests.post(url, json=wine_sample)
print(response.json())
```

### Expected Response

```json
{
  "prediction": 0,
  "class_name": "class_0",
  "probabilities": {
    "class_0": 0.92,
    "class_1": 0.05,
    "class_2": 0.03
  }
}
```

## FastAPI Features Used

### Request Body Reading
FastAPI automatically reads the request body as JSON when a client sends data to endpoints like `/predict`.

### Data Conversion
Pydantic models automatically convert JSON data to Python types. For example, if `"alcohol": "13.0"` is sent as a string, it will be converted to a float.

### Data Validation
Pydantic validates all incoming data:
- Checks required fields are present
- Validates data types
- Enforces constraints (e.g., `ge=0` for non-negative values)
- Returns 422 Unprocessable Entity if validation fails

### Error Handling
```python
from fastapi import HTTPException

raise HTTPException(status_code=500, detail="Error message")
```

The `HTTPException` class is used to return error responses with appropriate status codes.

## Additional Tools

- **Postman**: Can be used for API testing
- **ReDoc**: Alternative API documentation at http://127.0.0.1:8000/redoc

## Requirements

```
scikit-learn==1.5.1
fastapi[all]==0.111.1
numpy==1.26.4
joblib==1.4.2
```

## Author

Hitarth - MLOps Lab Assignment