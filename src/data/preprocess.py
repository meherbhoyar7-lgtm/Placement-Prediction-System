from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from src.data.load_data import load_data
import pandas as pd


def split_data(df, target_columns,drop_columns=None,stratify=False):
    if drop_columns is None:
        drop_columns = []
    X = df.drop(
        columns=drop_columns + [target_columns])
    y = df[target_columns]
    stratify_value = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value
    )
    return X_train, X_test, y_train, y_test

def split_X_data(df, drop_columns=None):
    if drop_columns is None:
        drop_columns = []

    X = df.drop(columns=drop_columns)
    return X

def handle_missing_values(X_train, X_test, numerical_features):
    imputer = SimpleImputer(strategy="median")
    X_train = X_train.copy()
    X_test = X_test.copy()
    # Fit only on training data
    X_train[numerical_features] = imputer.fit_transform(
        X_train[numerical_features]
    )
    X_test[numerical_features] = imputer.transform(
        X_test[numerical_features]
    )
    return X_train, X_test, imputer


def identify_features(X):
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    # FIXED: Added "string" to prevent the Pandas warning
    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    return numerical_features, categorical_features


def standardize_data(X_train, X_test, numerical_features):
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    # Fit only on training data
    X_train[numerical_features] = scaler.fit_transform(
        X_train[numerical_features]
    )
    # Use the same scaler for test data
    X_test[numerical_features] = scaler.transform(
        X_test[numerical_features]
    )
    return X_train, X_test, scaler


def one_hot_encode_data(X_train, X_test, categorical_features):
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit only on training data
    train_encoded = encoder.fit_transform(
        X_train[categorical_features]
    )

    # Use the same encoder for test data
    test_encoded = encoder.transform(
        X_test[categorical_features]
    )

    # Get encoded column names
    encoded_columns = encoder.get_feature_names_out(
        categorical_features
    )

    # Create DataFrames
    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=encoded_columns,
        index=X_train.index
    )
    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=encoded_columns,
        index=X_test.index
    )

    # Remove original categorical columns
    X_train = X_train.drop(
        columns=categorical_features
    )
    X_test = X_test.drop(
        columns=categorical_features
    )
    # Add encoded columns
    X_train = pd.concat(
        [X_train, train_encoded_df],
        axis=1
    )
    X_test = pd.concat(
        [X_test, test_encoded_df],
        axis=1
    )
    return X_train, X_test, encoder


def ordinal_encode_data(X_train, X_test, ordinal_features):
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1  # <--- CHANGED FROM 1 TO -1
    )
    X_train = X_train.copy()
    X_test = X_test.copy()

    train_encoded = ordinal_encoder.fit_transform(
        X_train[ordinal_features]
    )
    test_encoded = ordinal_encoder.transform(
        X_test[ordinal_features]
    )
    #Convert the Dataframe
    train_encoded_df = pd.DataFrame(
        train_encoded,
        columns=ordinal_features,
        index=X_train.index
    )
    test_encoded_df = pd.DataFrame(
        test_encoded,
        columns=ordinal_features,
        index=X_test.index
    )

    X_train = X_train.drop(
        columns=ordinal_features
    )
    X_test = X_test.drop(
        columns=ordinal_features
    )

    X_train = pd.concat(
        [X_train, train_encoded_df],
        axis=1
    )
    X_test = pd.concat(
        [X_test, test_encoded_df],
        axis=1
    )
    return X_train, X_test, ordinal_encoder

if __name__ == "__main__":
    df = load_data()
    print("Original Dataset Shape:")
    print(df.shape)
    X_train, X_test, y_train, y_test = split_data(df)
    print("\nTraining Dataset Shape:")
    print(X_train.shape)
    print("\nTest Dataset Shape:")
    print(X_test.shape)

    numerical_features, categorical_features = identify_features(X_train)
    print("Numerical Features:")
    print(numerical_features)
    print("\nCategorical Features:")
    print(categorical_features)

    # FIXED: Changed Specialization to Specialization (with an 's')
    one_hot_features = [
        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs"
    ]
    ordinal_features = [
        "CollegeTier",
        "CGPA_Tier"
    ]

    X_train, X_test, imputer = handle_missing_values(X_train, X_test, numerical_features)
    print(X_train[numerical_features].isnull().sum())
    print("\nMissing values handling completed.")

    X_train, X_test, scaler = standardize_data(
        X_train,
        X_test,
        numerical_features
    )
    print("\nStandardization completed.")

    # FIXED: Passed one_hot_features instead of categorical_features
    X_train, X_test, encoder = one_hot_encode_data(
        X_train,
        X_test,
        one_hot_features
    )
    print("\nOne-Hot Encoding completed.")

    X_train, X_test, ordinal_encoder = ordinal_encode_data(X_train, X_test, ordinal_features)
    print("\nOrdinal Encoding completed.")

    X_train["PlacementStatus"] = y_train
    X_test["PlacementStatus"] = y_test

    # Fixed the double "Data/Data/" typo in the second save path
    X_train.to_csv("../../Data/preprocessed_train.csv", index=False)
    X_test.to_csv("../../Data/preprocessed_test.csv", index=False)

    print("\nTraining Data:")
    print(X_train.head())

    print("\nTesting Data:")
    print(X_test.head())

    print("\nFinal Training Shape:")
    print(X_train.shape)

    print("\nFinal Testing Shape:")
    print(X_test.shape)


