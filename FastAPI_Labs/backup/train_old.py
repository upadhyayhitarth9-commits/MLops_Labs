from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from data import load_data, split_data, scale_data

def fit_model(X_train, y_train, n_estimators=100, max_depth=5):
    """
    Train a Random Forest Classifier.
    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
        n_estimators (int): Number of trees in the forest.
        max_depth (int): Maximum depth of trees.
    Returns:
        model: Trained Random Forest model.
    """
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

def evaluate_model(model, X_test, y_test, target_names):
    """
    Evaluate the model and print metrics.
    Args:
        model: Trained model.
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): Test target values.
        target_names (list): Names of target classes.
    Returns:
        dict: Evaluation metrics.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return {"accuracy": accuracy, "report": report}

def save_model(model, scaler, feature_names, target_names, metrics):
    """
    Save the model, scaler, and metadata.
    """
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
    # Load and prepare data
    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    # Train model
    print("Training Random Forest Classifier...")
    model = fit_model(X_train_scaled, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test_scaled, y_test, target_names)
    
    # Save everything
    save_model(model, scaler, feature_names, target_names, metrics)