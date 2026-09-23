import sys
import os
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Add the project root to the python path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data.load_data import load_data
from src.data.preprocess import (
    split_data, identify_features, handle_missing_values,
    standardize_data, one_hot_encode_data, ordinal_encode_data
)

def create_model(nu=0.035):
    """Create the One-Class SVM model.
    nu is an upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
    We set it to the expected anomaly rate.
    """
    return OneClassSVM(nu=nu, kernel="rbf", gamma="scale")

def train_model(model, X_train):
    """Train the model."""
    model.fit(X_train)
    return model

def predict_anomalies(model, X_test):
    """Predict anomalies. Output is 1 for normal, -1 for anomaly.
    Convert to 0 for normal, 1 for anomaly."""
    preds = model.predict(X_test)
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
    
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_columns="IsAnomaly",
        drop_columns=["StudentID", "PlacementStatus", "Salary Package"]
    )
    
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    
    # One-Class SVM can be slow on very large datasets. 
    # Subsampling for training speed if needed, but we'll try full data first.
    # To ensure it runs in a reasonable time, we'll subsample to 10000 rows for training
    print("Subsampling training data to 10,000 for faster SVM training...")
    # Get a stratified sample if possible, or just a random sample.
    X_train_sub = X_train.sample(n=10000, random_state=42)
    # Target doesn't strictly matter for training OCSVM since it's unsupervised, 
    # but we will just slice the corresponding indices if we needed to evaluate on train
    
    numerical_features, categorical_features = identify_features(X_train)
    one_hot_features = ['Gender', 'City', 'Stream', 'Specialisation', 'Hostel', 'HistoryOfBacklogs']
    ordinal_features = ['CollegeTier', 'CGPA_Tier']
    
    # Preprocess (fit on all train to maintain consistency)
    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    X_train, X_test, scaler = standardize_data(X_train, X_test, numerical_features)
    X_train, X_test, one_hot_encoder = one_hot_encode_data(X_train, X_test, one_hot_features)
    X_train, X_test, ordinal_encoder = ordinal_encode_data(X_train, X_test, ordinal_features)
    
    # Extract the subsampled transformed data
    X_train_sub_transformed = X_train.loc[X_train_sub.index]
    
    print("\nTraining One-Class SVM...")
    model = create_model()
    model = train_model(model, X_train_sub_transformed)
    
    print("Predicting and Evaluating...")
    y_pred = predict_anomalies(model, X_test)
    metrics = evaluate_model(y_test, y_pred)
    
    print("\n--- One-Class SVM Results ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} (Anomaly class)")
    print(f"Recall:    {metrics['recall']:.4f} (Anomaly class)")
    print(f"F1 Score:  {metrics['f1']:.4f} (Anomaly class)")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
