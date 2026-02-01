from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from data import load_data, split_data, scale_data

def fit_model(X_train, y_train, n_estimators=100, max_depth=5):
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

def evaluate_model(model, X_test, y_test, target_names):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    return {"accuracy": accuracy}

def save_model(model, scaler, feature_names, target_names, metrics):
    joblib.dump(model, "../model/wine_model.pkl")
    joblib.dump(scaler, "../model/scaler.pkl")
    metadata = {
        "feature_names": list(feature_names),
        "target_names": list(target_names),
        "model_type": "RandomForestClassifier",
        "accuracy": metrics["accuracy"]
    }
    with open("../model/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("\nModel, scaler, and metadata saved successfully!")

if __name__ == "__main__":
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    print("Training Random Forest Classifier...")
    model = fit_model(X_train_scaled, y_train)
    metrics = evaluate_model(model, X_test_scaled, y_test, target_names)
    save_model(model, scaler, feature_names, target_names, metrics)
