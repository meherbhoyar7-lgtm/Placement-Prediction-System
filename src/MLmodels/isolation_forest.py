import sys
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Add the project root to the python path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.load_data import load_data
from src.data.preprocess import (
    split_data, identify_features, handle_missing_values,
    standardize_data, one_hot_encode_data, ordinal_encode_data
)

def create_model(contamination=0.035):
    """Create the Isolation Forest model."""
    # contamination is the expected proportion of outliers (1750 / 50000 = 0.035)
    return IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)

def train_model(model, X_train):
    """Train the model."""
    model.fit(X_train)
    return model

def predict_anomalies(model, X_test):
    """Predict anomalies. Output is 1 for normal, -1 for anomaly.
    We convert this to 0 for normal, 1 for anomaly to match 'IsAnomaly' column."""
    preds = model.predict(X_test)
    # Convert from 1 (normal), -1 (anomaly) to 0 (normal), 1 (anomaly)
    converted_preds = [1 if p == -1 else 0 for p in preds]
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
    
    # We use IsAnomaly as the target just to evaluate our unsupervised model
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_columns="IsAnomaly",
        drop_columns=["StudentID", "PlacementStatus", "Salary Package"]
    )
    
    # Convert target to integers if they are not already
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
    
    print("\nTraining Isolation Forest...")
    model = create_model()
    model = train_model(model, X_train)
    
    print("Predicting and Evaluating...")
    y_pred = predict_anomalies(model, X_test)
    metrics = evaluate_model(y_test, y_pred)
    
    print("\n--- Isolation Forest Results ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} (Anomaly class)")
    print(f"Recall:    {metrics['recall']:.4f} (Anomaly class)")
    print(f"F1 Score:  {metrics['f1']:.4f} (Anomaly class)")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
