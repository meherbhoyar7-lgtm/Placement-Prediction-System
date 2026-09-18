import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    one_hot_encode_data,
    ordinal_encode_data,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

def create_model():
    model = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=42,
        oob_score=True,
    )
    # Removed model.fit(X_train, y_train) from here
    return model


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    print("\nRandom Forest Model Training Complete")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy)
    print("\nClassification Report :")
    print(classification_report(y_test, y_pred))
    return y_pred


def display_model(model, feature_names):
    plt.figure(figsize=(25, 12))

    # Extract the very first tree from the Random Forest
    first_tree = model.estimators_[0]

    tree.plot_tree(
        first_tree,
        feature_names=feature_names,
        class_names=["Not Placed", "Placed"],
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=4  # Limits depth so the plot is readable
    )
    plt.title("Random Forest (Tree 1 of 100) - Placement Prediction")
    plt.show()


def main():
    df = load_data()
    print("\nOriginal Dataset Shape:")
    print(df.shape)

    # Train-Test Split
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_columns="PlacementStatus",
        drop_columns=[
            "StudentID",
            "Salary Package",
            "IsAnomaly"
        ]
    )

    print("\nTraining Shape:")
    print(X_train.shape)
    print("\nTesting Shape:")
    print(X_test.shape)

    # Extract features using the function rather than importing static variables
    numerical_features, categorical_features = identify_features(X_train)

    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)

    one_hot_features = [
        'Gender', 'City', 'Stream', 'Specialisation', 'Hostel', 'HistoryOfBacklogs'
    ]
    ordinal_features = [
        'CollegeTier', 'CGPA_Tier'
    ]

    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    print("Missing values handled.")

    X_train, X_test, one_hot_encoder = one_hot_encode_data(X_train, X_test, one_hot_features)
    print("One-hot encoding completed.")

    X_train, X_test, ordinal_encoder = ordinal_encode_data(
        X_train,
        X_test,
        ordinal_features
    )
    print("Ordinal encoding completed.")

    model = create_model()
    model = train_model(model, X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)

    # Display the first tree from the trained Random Forest
    display_model(model, feature_names=X_train.columns)

    print("\nModel training completed.")


if __name__ == "__main__":
    main()