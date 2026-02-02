# main.py
# Wine Classification API using Flask and TensorFlow
# Modified from Iris Classifier for MLOps Lab Assignment

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__, static_folder='statics', template_folder='templates')

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model('wine_model.keras')
print("Model loaded successfully!")

# Load the scaler
print("Loading scaler...")
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
print("Scaler loaded successfully!")

# Wine class labels
CLASS_LABELS = ['Class 0 (Type A)', 'Class 1 (Type B)', 'Class 2 (Type C)']

# Feature names for the Wine dataset (13 features)
FEATURE_NAMES = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
    'color_intensity', 'hue', 'od280_od315', 'proline'
]


@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint for container orchestration"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })


@app.route('/info')
def info():
    """Return model and dataset information"""
    return jsonify({
        "application": "Wine Classifier API",
        "version": "1.0.0",
        "framework": "TensorFlow/Keras",
        "dataset": "UCI Wine Dataset",
        "classes": CLASS_LABELS,
        "features": FEATURE_NAMES,
        "num_features": len(FEATURE_NAMES),
        "author": "MLOps Lab Assignment"
    })


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction endpoint - handles both web form and API requests"""
    
    if request.method == 'GET':
        # Return the prediction form
        return render_template('predict.html', features=FEATURE_NAMES)
    
    elif request.method == 'POST':
        try:
            # Get data from request (supports both form and JSON)
            if request.is_json:
                data = request.get_json()
            else:
                data = request.form
            
            # Extract all 13 features
            features = []
            for feature_name in FEATURE_NAMES:
                if feature_name not in data:
                    return jsonify({
                        "error": f"Missing required feature: {feature_name}"
                    }), 400
                
                try:
                    value = float(data[feature_name])
                    features.append(value)
                except ValueError:
                    return jsonify({
                        "error": f"Invalid value for {feature_name}: must be a number"
                    }), 400
            
            # Prepare input data
            input_data = np.array(features).reshape(1, -1)
            
            # Scale the input using the same scaler from training
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled, verbose=0)
            predicted_class_idx = np.argmax(prediction)
            predicted_class = CLASS_LABELS[predicted_class_idx]
            confidence = float(np.max(prediction)) * 100
            
            # Prepare response
            response = {
                "success": True,
                "predicted_class": predicted_class,
                "predicted_class_index": int(predicted_class_idx),
                "confidence": f"{confidence:.2f}%",
                "probabilities": {
                    CLASS_LABELS[i]: f"{prediction[0][i]*100:.2f}%"
                    for i in range(len(CLASS_LABELS))
                },
                "input_features": {
                    FEATURE_NAMES[i]: features[i]
                    for i in range(len(FEATURE_NAMES))
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
    
    else:
        return jsonify({"error": "Method not allowed"}), 405


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🍷 Wine Classifier API")
    print("=" * 50)
    print(f"Model loaded with {len(FEATURE_NAMES)} features")
    print(f"Classes: {CLASS_LABELS}")
    print("\nEndpoints:")
    print("  GET  /         - Home page")
    print("  GET  /predict  - Prediction form")
    print("  POST /predict  - Make prediction")
    print("  GET  /health   - Health check")
    print("  GET  /info     - API information")
    print("\nStarting server on http://localhost:4000")
    print("=" * 50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=4000)