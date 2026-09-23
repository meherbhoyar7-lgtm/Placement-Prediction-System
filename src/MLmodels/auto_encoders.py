import sys
import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

# Add the project root to the python path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.load_data import load_data
from src.data.preprocess import (
    split_data, identify_features, handle_missing_values,
    standardize_data, one_hot_encode_data, ordinal_encode_data
)

def create_model(hidden_layer_sizes=(16, 8, 16)):
    """Create the Auto Encoder model using MLPRegressor.
    An autoencoder attempts to reconstruct the input.
    """
    return MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        max_iter=100, # Keep it small for fast training
        random_state=42,
        early_stopping=True
    )

def train_model(model, X_train):
    """Train the model to reconstruct its input."""
    # Target is the input itself
    model.fit(X_train, X_train)
    return model

def predict_anomalies(model, X_test, threshold=None):
    """Predict anomalies based on reconstruction error."""
    # Reconstruct the inputs
    X_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error per sample
    mse = np.mean(np.power(X_test - X_pred, 2), axis=1)
    
    # If no threshold is provided, use the 96.5th percentile (for 3.5% contamination)
    if threshold is None:
        threshold = np.percentile(mse, 96.5)
        
    # Anomaly if MSE > threshold (1 for anomaly, 0 for normal)
    converted_preds = (mse > threshold).astype(int)
    return converted_preds

def evaluate_model(y_true, y_pred):
    """Evaluate the model using classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    df = load_data()
    print("Original Dataset Shape:", df.shape)
    
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_columns="IsAnomaly",
        drop_columns=["StudentID", "PlacementStatus", "Salary Package"]
    )
    
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    numerical_features, categorical_features = identify_features(X_train)
    one_hot_features = ['Gender', 'City', 'Stream', 'Specialisation', 'Hostel', 'HistoryOfBacklogs']
    ordinal_features = ['CollegeTier', 'CGPA_Tier']
    
    # Preprocess
    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    X_train, X_test, scaler = standardize_data(X_train, X_test, numerical_features)
    X_train, X_test, one_hot_encoder = one_hot_encode_data(X_train, X_test, one_hot_features)
    X_train, X_test, ordinal_encoder = ordinal_encode_data(X_train, X_test, ordinal_features)
    
    print("\nTraining Auto Encoder (MLPRegressor)...")
    model = create_model()
    model = train_model(model, X_train)
    
    print("Predicting and Evaluating...")
    y_pred = predict_anomalies(model, X_test)
    metrics = evaluate_model(y_test, y_pred)
    
    print("\n--- Auto Encoder Results ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} (Anomaly class)")
    print(f"Recall:    {metrics['recall']:.4f} (Anomaly class)")
    print(f"F1 Score:  {metrics['f1']:.4f} (Anomaly class)")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
