# 🏠 House Price Prediction API

An ML-powered API that predicts house prices based on property features, built with FastAPI and scikit-learn.

## Features

- **Machine Learning Model**: Random Forest regression trained on synthetic housing data
- **Interactive Web UI**: Beautiful, responsive interface for predictions
- **RESTful API**: Programmatic access for integrations
- **Dockerized**: Easy deployment with Docker
- **Price Range Estimates**: Get low and high estimates (±10%)

## Tech Stack

- **Framework**: FastAPI
- **Server**: Uvicorn
- **ML Library**: scikit-learn (Random Forest)
- **Data Processing**: pandas, NumPy
- **Model Serialization**: joblib
- **Containerization**: Docker

## Input Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `square_feet` | float | > 0 | Property size in sq ft |
| `bedrooms` | int | ≥ 0 | Number of bedrooms |
| `bathrooms` | int | ≥ 0 | Number of bathrooms |
| `age_years` | int | ≥ 0 | Property age in years |
| `garage_size` | int | 0-4 | Garage capacity (cars) |
| `has_pool` | bool | - | Has swimming pool |
| `has_garden` | bool | - | Has garden/yard |
| `location_score` | float | 1-10 | Location quality rating |

## Quick Start

### Using Docker (Recommended)

```bash
# Build the image
docker build -t house-price-api .

# Run the container
docker run -p 8000:8000 house-price-api
```

Open **http://localhost:8000** in your browser.

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train_model.py

# Run the server
uvicorn main:app --reload --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI interface |
| `GET` | `/api` | API information |
| `POST` | `/predict` | Get price prediction |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger documentation |

## API Usage Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "square_feet": 1800,
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 10,
    "garage_size": 2,
    "has_pool": false,
    "has_garden": true,
    "location_score": 7.5
  }'
```

### Response

```json
{
  "predicted_price": 425000.00,
  "price_range": {
    "low": 382500.00,
    "high": 467500.00
  },
  "input_features": {
    "square_feet": 1800,
    "bedrooms": 3,
    "bathrooms": 2,
    "age_years": 10,
    "garage_size": 2,
    "has_pool": false,
    "has_garden": true,
    "location_score": 7.5
  }
}
```

## Project Structure

```
House-price-api/
├── main.py           # FastAPI application with Web UI
├── train_model.py    # Model training script
├── requirements.txt  # Python dependencies
├── Dockerfile        # Docker configuration
└── model/
    └── house_price_model.pkl  # Trained model (generated)
```

## Model Details

The Random Forest model is trained on synthetic housing data with the following price factors:

- **$150** per square foot
- **$15,000** per bedroom
- **$10,000** per bathroom
- **$1,000** per year (newer = more expensive)
- **$8,000** per garage spot
- **$25,000** for pool
- **$10,000** for garden
- **$20,000** × location score

## Screenshots

### Web Interface
Access the beautiful prediction UI at `http://localhost:8000`

### API Documentation
Interactive Swagger docs available at `http://localhost:8000/docs`

## License

MIT License
