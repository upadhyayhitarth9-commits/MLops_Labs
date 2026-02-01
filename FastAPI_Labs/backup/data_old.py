import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Load the Wine dataset and return the features, target values, and feature names.
    Returns:
        X (numpy.ndarray): The features of the Wine dataset.
        y (numpy.ndarray): The target values of the Wine dataset.
        feature_names (list): Names of the features.
        target_names (list): Names of the target classes.
    """
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    return X, y, feature_names, target_names

def split_data(X, y, test_size=0.25, random_state=42):
    """
    Split the data into training and testing sets.
    Args:
        X (numpy.ndarray): The features of the dataset.
        y (numpy.ndarray): The target values of the dataset.
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random seed for reproducibility.
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test):
    """
    Scale features using StandardScaler.
    Args:
        X_train (numpy.ndarray): Training features.
        X_test (numpy.ndarray): Testing features.
    Returns:
        X_train_scaled, X_test_scaled, scaler: Scaled data and fitted scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

if __name__ == "__main__":
    X, y, feature_names, target_names = load_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Classes: {target_names}")
    print(f"Class distribution: {np.bincount(y)}")