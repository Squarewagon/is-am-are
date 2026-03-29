"""
Neural Network (MLP) utilities for Iris species classification.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


SPECIES_NAMES = ["setosa", "versicolor", "virginica"]


def load_iris_data():
    """Load the raw (imperfect) Iris dataset."""
    df = pd.read_csv("data/iris_raw.csv")
    return df


def preprocess_iris(df):
    """
    Preprocess the imperfect Iris dataset.

    Steps:
    1. Remove duplicate rows
    2. Remove obvious outliers (sepal_length > 10 or petal_width < 0)
    3. Impute missing values with column median
    4. Encode target labels
    5. Scale features with StandardScaler

    Returns X_train, X_test, y_train, y_test, scaler, label_encoder, feature_names
    """
    data = df.copy()
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    initial_shape = data.shape

    # Step 1: Remove duplicate rows
    data.drop_duplicates(inplace=True)

    # Step 2: Remove outliers (threshold 10.0 cm chosen as >25% above known max of 7.9 cm)
    data = data[data["sepal_length"].isna() | (data["sepal_length"] <= 10.0)]
    data = data[data["petal_width"].isna() | (data["petal_width"] >= 0.0)]

    # Step 3: Impute missing values with median
    for col in feature_cols:
        col_median = data[col].median()
        data[col] = data[col].fillna(col_median)

    # Step 4: Encode target labels using the integer 'species' column (0=setosa, 1=versicolor, 2=virginica)
    le = LabelEncoder()
    y = le.fit_transform(data["species"].values)
    X = data[feature_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 5: Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le, feature_cols


@st.cache_resource
def get_nn_model():
    """
    Train and cache the MLP Neural Network on the Iris dataset.

    Network architecture:
      Input  : 4 features
      Layer 1: 64 neurons, ReLU activation
      Layer 2: 32 neurons, ReLU activation
      Layer 3: 16 neurons, ReLU activation
      Output : 3 classes (Softmax via sklearn cross-entropy loss)

    Returns: (model, scaler, label_encoder, accuracy, report, cm)
    """
    df = load_iris_data()
    X_train, X_test, y_train, y_test, scaler, le, feature_cols = preprocess_iris(df)

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
        learning_rate_init=0.001,
        n_iter_no_change=20,
        tol=1e-5,
    )
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=SPECIES_NAMES, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return mlp, scaler, le, accuracy, report, cm


def predict_species(model, scaler, sepal_length, sepal_width, petal_length, petal_width):
    """
    Predict Iris species for a single flower.

    Returns
    -------
    prediction : int   (0=setosa, 1=versicolor, 2=virginica)
    species_name: str
    probability : list [p_setosa, p_versicolor, p_virginica]
    """
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0].tolist()
    species_name = SPECIES_NAMES[prediction]

    return int(prediction), species_name, probability