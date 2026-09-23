import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import pandas as pd
import joblib
from flask import Flask, render_template, request, send_from_directory
from src.data.load_data import load_data, get_summary

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "Data")
MODELS_DIR = os.path.join(BASE_DIR, "Models")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dataset")
def dataset():
    df = load_data()
    summary = get_summary(df)
    return render_template(
        "load_dataset.html",
        summary=summary,
        first_rows=df.head().to_html(index=False, classes="data-table")
    )


@app.route("/eda")
def eda():
    return render_template("eda.html")


@app.route("/preprocessing")
def preprocessing():
    train_path = os.path.join(DATA_DIR, "preprocessed_train.csv")
    test_path = os.path.join(DATA_DIR, "preprocessed_test.csv")

    file_exists = os.path.exists(train_path)
    shape = (0, 0)
    train_head = ""

    if file_exists:
        train_df = pd.read_csv(train_path)
        shape = train_df.shape
        train_head = train_df.head(10).to_html(
            index=False, classes="data-table"
        )

    return render_template(
        "preprocessing.html",
        file_exists=file_exists,
        shape=shape,
        train_head=train_head,
    )


@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(DATA_DIR, filename, as_attachment=True)


@app.route("/models")
def models():
    train_path = os.path.join(DATA_DIR, "preprocessed_train.csv")
    test_path = os.path.join(DATA_DIR, "preprocessed_test.csv")

    classification_results = []
    linear_results = []
    clustering_results = []
    anomaly_results = []
    data_available = os.path.exists(train_path) and os.path.exists(test_path)

    if data_available:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # --- Classification Models ---
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        X_train_cls = train_df.drop(columns="PlacementStatus")
        y_train_cls = train_df["PlacementStatus"]
        X_test_cls = test_df.drop(columns="PlacementStatus")
        y_test_cls = test_df["PlacementStatus"]

        cls_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.5, random_state=42)
        }

        classification_results = []
        for name, model in cls_models.items():
            model.fit(X_train_cls, y_train_cls)
            y_pred_cls = model.predict(X_test_cls)
            classification_results.append({
                "name": name,
                "accuracy": round(accuracy_score(y_test_cls, y_pred_cls) * 100, 2),
                "precision": round(precision_score(y_test_cls, y_pred_cls, zero_division=0) * 100, 2),
                "recall": round(recall_score(y_test_cls, y_pred_cls, zero_division=0) * 100, 2),
                "f1": round(f1_score(y_test_cls, y_pred_cls, zero_division=0) * 100, 2),
            })

        # --- Linear Regression (Salary Prediction) ---
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # For linear regression, reload original data to get Salary Package
        from src.data.preprocess import (
            split_data, identify_features, handle_missing_values,
            standardize_data, one_hot_encode_data, ordinal_encode_data
        )

        raw_df = load_data()
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(
            raw_df,
            target_columns="Salary Package",
            drop_columns=["StudentID", "PlacementStatus", "IsAnomaly"]
        )
        num_feats, cat_feats = identify_features(X_train_reg)
        one_hot_feats = [
            'Gender', 'City', 'Stream', 'Specialisation',
            'Hostel', 'HistoryOfBacklogs'
        ]
        ordinal_feats = ['CollegeTier', 'CGPA_Tier']

        X_train_reg, X_test_reg, _ = handle_missing_values(
            X_train_reg, X_test_reg, num_feats
        )
        X_train_reg, X_test_reg, _ = standardize_data(
            X_train_reg, X_test_reg, num_feats
        )
        X_train_reg, X_test_reg, _ = one_hot_encode_data(
            X_train_reg, X_test_reg, one_hot_feats
        )
        X_train_reg, X_test_reg, _ = ordinal_encode_data(
            X_train_reg, X_test_reg, ordinal_feats
        )

        reg_models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.01),
            "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
        }

        linear_results = []
        for name, model in reg_models.items():
            model.fit(X_train_reg, y_train_reg)
            y_pred_reg = model.predict(X_test_reg)
            linear_results.append({
                "name": name,
                "mae": round(mean_absolute_error(y_test_reg, y_pred_reg), 2),
                "mse": round(mean_squared_error(y_test_reg, y_pred_reg), 2),
                "rmse": round(mean_squared_error(y_test_reg, y_pred_reg) ** 0.5, 2),
                "r2": round(r2_score(y_test_reg, y_pred_reg) * 100, 2),
            })

        # --- Clustering Models ---
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        
        # Sample the preprocessed data to avoid memory issues and speed up hierarchy clustering
        X_clust_sample = X_train_cls.sample(n=500, random_state=42)
        
        kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=42)
        labels_kmeans = kmeans.fit_predict(X_clust_sample)
        score_kmeans = silhouette_score(X_clust_sample, labels_kmeans)
        
        agglo = AgglomerativeClustering(n_clusters=3, linkage="ward")
        labels_agglo = agglo.fit_predict(X_clust_sample)
        score_agglo = silhouette_score(X_clust_sample, labels_agglo)
        
        clustering_results = [
            {"name": "K-Means Clustering", "silhouette": round(score_kmeans, 4)},
            {"name": "Agglomerative Clustering", "silhouette": round(score_agglo, 4)}
        ]

        # --- Anomaly Detection Models ---
        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM
        from sklearn.neural_network import MLPRegressor
        import numpy as np

        # For anomaly detection we'll use a sample to keep page load times fast
        X_train_anomaly = X_train_reg.sample(n=2000, random_state=42)
        X_test_anomaly = X_test_reg.sample(n=500, random_state=42)
        y_test_anomaly = raw_df.loc[X_test_anomaly.index, 'IsAnomaly'].astype(int)

        # 1. Isolation Forest
        iso_forest = IsolationForest(contamination=0.035, random_state=42)
        iso_forest.fit(X_train_anomaly)
        preds_iso = iso_forest.predict(X_test_anomaly)
        preds_iso_converted = [1 if p == -1 else 0 for p in preds_iso]
        
        # 2. One-Class SVM
        oc_svm = OneClassSVM(nu=0.035, kernel="rbf", gamma="scale")
        oc_svm.fit(X_train_anomaly)
        preds_svm = oc_svm.predict(X_test_anomaly)
        preds_svm_converted = [1 if p == -1 else 0 for p in preds_svm]
        
        # 3. Auto Encoder
        ae = MLPRegressor(hidden_layer_sizes=(16, 8, 16), activation='relu', solver='adam', max_iter=100, random_state=42)
        ae.fit(X_train_anomaly, X_train_anomaly)
        ae_preds = ae.predict(X_test_anomaly)
        mse = np.mean(np.power(X_test_anomaly - ae_preds, 2), axis=1)
        threshold = np.percentile(mse, 96.5)
        preds_ae_converted = (mse > threshold).astype(int)

        anomaly_results = [
            {
                "name": "Isolation Forest",
                "accuracy": round(accuracy_score(y_test_anomaly, preds_iso_converted) * 100, 2),
                "f1": round(f1_score(y_test_anomaly, preds_iso_converted, zero_division=0) * 100, 2)
            },
            {
                "name": "One-Class SVM",
                "accuracy": round(accuracy_score(y_test_anomaly, preds_svm_converted) * 100, 2),
                "f1": round(f1_score(y_test_anomaly, preds_svm_converted, zero_division=0) * 100, 2)
            },
            {
                "name": "Auto Encoder",
                "accuracy": round(accuracy_score(y_test_anomaly, preds_ae_converted) * 100, 2),
                "f1": round(f1_score(y_test_anomaly, preds_ae_converted, zero_division=0) * 100, 2)
            }
        ]

    return render_template(
        "models.html",
        data_available=data_available,
        classification_results=classification_results,
        linear_results=linear_results,
        clustering_results=clustering_results,
        anomaly_results=anomaly_results,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5005)