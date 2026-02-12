"""
Train a house price prediction model.
This script generates synthetic data and trains a Random Forest model.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def generate_house_data(n_samples=1000):
    """Generate synthetic house data for training"""
    np.random.seed(42)
    
    # Generate features
    data = {
        'square_feet': np.random.randint(500, 5000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randls ~/MLops_Labs/Lab2/House-price-api/int(1, 4, n_samples),
        'age_years': np.random.randint(0, 50, n_samples),
        'garage_size': np.random.randint(0, 4, n_samples),
        'has_pool': np.random.randint(0, 2, n_samples),
        'has_garden': np.random.randint(0, 2, n_samples),
        'location_score': np.round(np.random.uniform(1, 10, n_samples), 1)
    }
    
    df = pd.DataFrame(data)
    
    # Generate realistic price based on features
    # Base price formula (simplified real-world factors)
    df['price'] = (
        df['square_feet'] * 150 +                    # $150 per sq ft base
        df['bedrooms'] * 15000 +                     # $15k per bedroom
        df['bathrooms'] * 10000 +                    # $10k per bathroom
        (50 - df['age_years']) * 1000 +              # Newer = more expensive
        df['garage_size'] * 8000 +                   # $8k per garage spot
        df['has_pool'] * 25000 +                     # $25k for pool
        df['has_garden'] * 10000 +                   # $10k for garden
        df['location_score'] * 20000 +               # Location multiplier
        np.random.normal(0, 20000, n_samples)        # Random noise
    )
    
    # Ensure no negative prices
    df['price'] = df['price'].clip(lower=50000)
    
    return df


def train_model():
    """Train and save the model"""
    print("=" * 50)
    print("🏠 House Price Model Training")
    print("=" * 50)
    
    # Generate data
    print("\n📊 Generating training data...")
    df = generate_house_data(1000)
    print(f"   Generated {len(df)} samples")
    
    # Show sample data
    print("\n📋 Sample data:")
    print(df.head().to_string())
    
    # Show price statistics
    print(f"\n💰 Price Statistics:")
    print(f"   Min:    ${df['price'].min():,.0f}")
    print(f"   Max:    ${df['price'].max():,.0f}")
    print(f"   Mean:   ${df['price'].mean():,.0f}")
    print(f"   Median: ${df['price'].median():,.0f}")
    
    # Prepare features and target
    feature_columns = [
        'square_feet', 'bedrooms', 'bathrooms', 'age_years',
        'garage_size', 'has_pool', 'has_garden', 'location_score'
    ]
    X = df[feature_columns]
    y = df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n🔀 Split data: {len(X_train)} train, {len(X_test)} test")
    
    # Train model
    print("\n🤖 Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("   Training complete!")
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 Model Performance:")
    print(f"   Mean Absolute Error: ${mae:,.0f}")
    print(f"   R² Score: {r2:.4f}")
    
    # Feature importance
    print(f"\n🎯 Feature Importance:")
    importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"   {row['feature']:15} {bar} {row['importance']:.3f}")
    
    # Save model
    os.makedirs('model', exist_ok=True)
    model_path = 'model/house_price_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Test prediction
    print("\n🧪 Test Prediction:")
    test_house = np.array([[1800, 3, 2, 15, 2, 0, 1, 7.5]])
    test_price = model.predict(test_house)[0]
    print(f"   House: 1800 sqft, 3 bed, 2 bath, 15 years old")
    print(f"   Predicted Price: ${test_price:,.0f}")
    
    print("\n" + "=" * 50)
    print("✅ Training complete! Model ready for predictions.")
    print("=" * 50)
    
    return model


if __name__ == "__main__":
    train_model()