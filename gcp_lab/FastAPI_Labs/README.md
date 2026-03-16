# Wine Quality Classification API

A machine learning API built with FastAPI that predicts wine classes using a Random Forest Classifier trained on the classic Wine dataset.

## Overview

This project provides a RESTful API for wine classification, featuring:
- Single and batch prediction endpoints
- Model information and feature importance retrieval
- Health check monitoring
- Comprehensive input validation using Pydantic

## Project Structure

```
.
├── src/
│   ├── app.py              # FastAPI application with endpoints
│   ├── data.py             # Data loading, splitting, and preprocessing
│   ├── predict.py          # Prediction functions and model loading
│   └── train.py            # Model training script
├── model/
│   ├── wine_model.pkl      # Trained Random Forest model
│   ├── scaler.pkl          # Fitted StandardScaler
│   └── metadata.json       # Model metadata and metrics
├── requirements.txt
├── Dockerfile
└── README.md
```

## Dataset

The API uses the **UCI Wine Dataset** containing 178 samples with 13 features:

| Feature | Description |
|---------|-------------|
| alcohol | Alcohol content |
| malic_acid | Malic acid content |
| ash | Ash content |
| alcalinity_of_ash | Alcalinity of ash |
| magnesium | Magnesium content |
| total_phenols | Total phenols |
| flavanoids | Flavanoids content |
| nonflavanoid_phenols | Nonflavanoid phenols |
| proanthocyanins | Proanthocyanins content |
| color_intensity | Color intensity |
| hue | Hue value |
| od280_od315 | OD280/OD315 of diluted wines |
| proline | Proline content |

**Target Classes:** 3 wine cultivars (class_0, class_1, class_2)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd wine-classification-api
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Loading and Preprocessing (`data.py`)

```python
from data import load_data, split_data, scale_data

# Load the Wine dataset
X, y, feature_names, target_names = load_data()
print(f"Dataset shape: {X.shape}")
print(f"Features: {feature_names}")
print(f"Classes: {target_names}")

# Split into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=42)

# Scale features using StandardScaler
X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
```

**Functions:**
- `load_data()` - Load Wine dataset, returns features, targets, and names
- `split_data(X, y, test_size, random_state)` - Split data with stratification
- `scale_data(X_train, X_test)` - Standardize features using StandardScaler

### Model Training (`train.py`)

```python
from train import fit_model, evaluate_model, save_model

# Train Random Forest Classifier
model = fit_model(X_train_scaled, y_train, n_estimators=100, max_depth=5)

# Evaluate model performance
metrics = evaluate_model(model, X_test_scaled, y_test, target_names)
# Output:
# Model Accuracy: 0.9778
# Classification Report:
#               precision    recall  f1-score   support
#      class_0       1.00      1.00      1.00        14
#      class_1       0.94      1.00      0.97        17
#      class_2       1.00      0.92      0.96        14

# Save model, scaler, and metadata
save_model(model, scaler, feature_names, target_names, metrics)
```

**Functions:**
- `fit_model(X_train, y_train, n_estimators, max_depth)` - Train Random Forest model
- `evaluate_model(model, X_test, y_test, target_names)` - Evaluate and print metrics
- `save_model(model, scaler, feature_names, target_names, metrics)` - Save artifacts to `model/` directory

**Run training:**
```bash
cd src
python train.py
```

### Making Predictions (`predict.py`)

```python
from predict import predict_data, predict_proba, get_feature_importance, get_model_info

# Single prediction
features = [[13.0, 2.0, 2.5, 19.0, 100.0, 2.5, 2.5, 0.3, 1.5, 5.0, 1.0, 3.0, 1000.0]]
prediction = predict_data(features)
probabilities = predict_proba(features)

# Get model information
info = get_model_info()
importance = get_feature_importance()
```

### Start the API Server (`app.py`)

```bash
cd src
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Access the API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check endpoint |
| GET | `/model/info` | Get model metadata and metrics |
| GET | `/model/features` | Get feature importance scores |
| POST | `/predict` | Predict wine class for single sample |
| POST | `/predict/batch` | Predict wine classes for multiple samples |

### Single Prediction Example

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
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

**Response:**
```json
{
  "prediction": 0,
  "class_name": "class_0",
  "probabilities": {
    "class_0": 0.85,
    "class_1": 0.10,
    "class_2": 0.05
  }
}
```

## Docker Support

**Build the image:**
```bash
docker build -t wine-classification-api .
```

**Run the container:**
```bash
docker run -p 8000:8000 wine-classification-api
```

## Requirements

- Python 3.9+
- FastAPI
- Uvicorn
- Scikit-learn
- NumPy
- Pydantic
- Joblib

## License

MIT License