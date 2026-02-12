from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="House Price Prediction API",
    description="Predict house prices based on features",
    version="1.0.0"
)

MODEL_PATH = "model/house_price_model.pkl"

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")


class HouseFeatures(BaseModel):
    square_feet: float = Field(..., gt=0)
    bedrooms: int = Field(..., ge=0)
    bathrooms: int = Field(..., ge=0)
    age_years: int = Field(..., ge=0)
    garage_size: int = Field(..., ge=0, le=4)
    has_pool: bool = False
    has_garden: bool = False
    location_score: float = Field(..., ge=1, le=10)


# ==================== HTML UI ====================
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the main UI"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏠 House Price Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #fff;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            padding: 40px 0;
        }
        
        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #a0a0a0;
            font-size: 1.1rem;
        }
        
        .card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 30px;
            margin-bottom: 20px;
        }
        
        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        .form-group {
            display: flex;
            flex-direction: column;
        }
        
        label {
            font-size: 0.85rem;
            color: #a0a0a0;
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        input[type="number"],
        input[type="range"] {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 12px 15px;
            color: #fff;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        input[type="number"]:focus {
            outline: none;
            border-color: #00d9ff;
            box-shadow: 0 0 20px rgba(0, 217, 255, 0.2);
        }
        
        input[type="range"] {
            padding: 5px;
            cursor: pointer;
        }
        
        .range-value {
            text-align: center;
            font-size: 1.5rem;
            font-weight: 600;
            color: #00d9ff;
            margin-top: 5px;
        }
        
        .checkbox-group {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .checkbox-item {
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(255, 255, 255, 0.05);
            padding: 15px 20px;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 2px solid transparent;
        }
        
        .checkbox-item:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        
        .checkbox-item.active {
            border-color: #00ff88;
            background: rgba(0, 255, 136, 0.1);
        }
        
        .checkbox-item input {
            display: none;
        }
        
        .checkbox-icon {
            font-size: 1.5rem;
        }
        
        .checkbox-label {
            font-weight: 500;
        }
        
        .btn {
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            border: none;
            padding: 15px 40px;
            font-size: 1.1rem;
            font-weight: 600;
            color: #1a1a2e;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 10px;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 217, 255, 0.3);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .result-card {
            display: none;
            text-align: center;
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.1), rgba(0, 255, 136, 0.1));
            border: 1px solid rgba(0, 255, 136, 0.3);
        }
        
        .result-card.show {
            display: block;
            animation: slideUp 0.5s ease;
        }
        
        @keyframes slideUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .price-label {
            font-size: 1rem;
            color: #a0a0a0;
            margin-bottom: 10px;
        }
        
        .price-value {
            font-size: 3.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 20px;
        }
        
        .price-range {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin-top: 20px;
        }
        
        .range-item {
            text-align: center;
        }
        
        .range-label {
            font-size: 0.8rem;
            color: #a0a0a0;
            margin-bottom: 5px;
        }
        
        .range-price {
            font-size: 1.3rem;
            font-weight: 600;
        }
        
        .range-price.low {
            color: #ff6b6b;
        }
        
        .range-price.high {
            color: #00ff88;
        }
        
        .features-summary {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin-top: 25px;
            padding-top: 25px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .feature-tag {
            background: rgba(255, 255, 255, 0.1);
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.85rem;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-top-color: #00d9ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .error-message {
            background: rgba(255, 107, 107, 0.1);
            border: 1px solid rgba(255, 107, 107, 0.3);
            color: #ff6b6b;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
            display: none;
        }
        
        .error-message.show {
            display: block;
        }
        
        footer {
            text-align: center;
            padding: 30px;
            color: #606060;
            font-size: 0.9rem;
        }
        
        footer a {
            color: #00d9ff;
            text-decoration: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏠 House Price Predictor</h1>
            <p class="subtitle">AI-powered real estate valuation using Machine Learning</p>
        </header>
        
        <div class="card">
            <h2 class="card-title">📐 Property Details</h2>
            <div class="form-grid">
                <div class="form-group">
                    <label>Square Feet</label>
                    <input type="number" id="square_feet" value="1800" min="100" max="10000">
                </div>
                <div class="form-group">
                    <label>Bedrooms</label>
                    <input type="number" id="bedrooms" value="3" min="0" max="10">
                </div>
                <div class="form-group">
                    <label>Bathrooms</label>
                    <input type="number" id="bathrooms" value="2" min="0" max="10">
                </div>
                <div class="form-group">
                    <label>Age (Years)</label>
                    <input type="number" id="age_years" value="10" min="0" max="100">
                </div>
                <div class="form-group">
                    <label>Garage Size (Cars)</label>
                    <input type="number" id="garage_size" value="2" min="0" max="4">
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2 class="card-title">📍 Location Score</h2>
            <input type="range" id="location_score" min="1" max="10" step="0.5" value="7" style="width: 100%">
            <div class="range-value"><span id="location_display">7.0</span> / 10</div>
        </div>
        
        <div class="card">
            <h2 class="card-title">✨ Amenities</h2>
            <div class="checkbox-group">
                <div class="checkbox-item" onclick="toggleCheckbox('has_pool', this)">
                    <input type="checkbox" id="has_pool">
                    <span class="checkbox-icon">🏊</span>
                    <span class="checkbox-label">Swimming Pool</span>
                </div>
                <div class="checkbox-item" onclick="toggleCheckbox('has_garden', this)">
                    <input type="checkbox" id="has_garden">
                    <span class="checkbox-icon">🌳</span>
                    <span class="checkbox-label">Garden</span>
                </div>
            </div>
        </div>
        
        <button class="btn" onclick="predictPrice()">
            🔮 Predict House Price
        </button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing property data...</p>
        </div>
        
        <div class="error-message" id="error"></div>
        
        <div class="card result-card" id="result">
            <p class="price-label">Estimated Market Value</p>
            <div class="price-value" id="predicted_price">$0</div>
            <div class="price-range">
                <div class="range-item">
                    <p class="range-label">Low Estimate</p>
                    <p class="range-price low" id="price_low">$0</p>
                </div>
                <div class="range-item">
                    <p class="range-label">High Estimate</p>
                    <p class="range-price high" id="price_high">$0</p>
                </div>
            </div>
            <div class="features-summary" id="features_summary"></div>
        </div>
        
        <footer>
            <p>Built with FastAPI + scikit-learn | <a href="/docs">API Documentation</a></p>
        </footer>
    </div>
    
    <script>
        // Update location score display
        const locationSlider = document.getElementById('location_score');
        const locationDisplay = document.getElementById('location_display');
        locationSlider.addEventListener('input', () => {
            locationDisplay.textContent = parseFloat(locationSlider.value).toFixed(1);
        });
        
        // Toggle checkbox
        function toggleCheckbox(id, element) {
            const checkbox = document.getElementById(id);
            checkbox.checked = !checkbox.checked;
            element.classList.toggle('active', checkbox.checked);
        }
        
        // Format currency
        function formatCurrency(amount) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                maximumFractionDigits: 0
            }).format(amount);
        }
        
        // Predict price
        async function predictPrice() {
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const error = document.getElementById('error');
            
            // Hide previous results
            result.classList.remove('show');
            error.classList.remove('show');
            loading.classList.add('show');
            
            // Gather form data
            const data = {
                square_feet: parseFloat(document.getElementById('square_feet').value),
                bedrooms: parseInt(document.getElementById('bedrooms').value),
                bathrooms: parseInt(document.getElementById('bathrooms').value),
                age_years: parseInt(document.getElementById('age_years').value),
                garage_size: parseInt(document.getElementById('garage_size').value),
                has_pool: document.getElementById('has_pool').checked,
                has_garden: document.getElementById('has_garden').checked,
                location_score: parseFloat(document.getElementById('location_score').value)
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                if (!response.ok) {
                    throw new Error('Prediction failed');
                }
                
                const prediction = await response.json();
                
                // Update UI
                document.getElementById('predicted_price').textContent = 
                    formatCurrency(prediction.predicted_price);
                document.getElementById('price_low').textContent = 
                    formatCurrency(prediction.price_range.low);
                document.getElementById('price_high').textContent = 
                    formatCurrency(prediction.price_range.high);
                
                // Feature summary
                const features = prediction.input_features;
                const summary = document.getElementById('features_summary');
                summary.innerHTML = `
                    <span class="feature-tag">📐 ${features.square_feet} sq ft</span>
                    <span class="feature-tag">🛏️ ${features.bedrooms} beds</span>
                    <span class="feature-tag">🚿 ${features.bathrooms} baths</span>
                    <span class="feature-tag">📅 ${features.age_years} years old</span>
                    <span class="feature-tag">🚗 ${features.garage_size} car garage</span>
                    ${features.has_pool ? '<span class="feature-tag">🏊 Pool</span>' : ''}
                    ${features.has_garden ? '<span class="feature-tag">🌳 Garden</span>' : ''}
                    <span class="feature-tag">📍 Location: ${features.location_score}/10</span>
                `;
                
                loading.classList.remove('show');
                result.classList.add('show');
                
            } catch (err) {
                loading.classList.remove('show');
                error.textContent = '❌ Error: Could not get prediction. Please try again.';
                error.classList.add('show');
            }
        }
    </script>
</body>
</html>
"""


# ==================== API ENDPOINTS ====================
@app.get("/api")
def api_info():
    return {
        "message": "House Price Prediction API",
        "endpoints": {
            "GET /": "Web UI",
            "POST /predict": "Predict house price",
            "GET /docs": "API documentation"
        }
    }


def prepare_features(house: HouseFeatures) -> np.ndarray:
    return np.array([[
        house.square_feet,
        house.bedrooms,
        house.bathrooms,
        house.age_years,
        house.garage_size,
        int(house.has_pool),
        int(house.has_garden),
        house.location_score
    ]])


@app.post("/predict")
def predict_price(house: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    features = prepare_features(house)
    predicted_price = model.predict(features)[0]
    
    return {
        "predicted_price": round(predicted_price, 2),
        "price_range": {
            "low": round(predicted_price * 0.9, 2),
            "high": round(predicted_price * 1.1, 2)
        },
        "input_features": house.model_dump()
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)