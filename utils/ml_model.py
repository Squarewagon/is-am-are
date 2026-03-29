"""
Ensemble ML model utilities for Titanic survival prediction.
"""

import warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_titanic_data():
    """Load the raw Titanic dataset."""
    df = pd.read_csv("data/titanic_raw.csv")
    return df


def preprocess_titanic(df):
    """
    Preprocess the Titanic dataset.

    Steps:
    1. Drop high-missing column (deck)
    2. Impute missing Age with median
    3. Impute missing Embarked with mode
    4. Encode categorical features (sex, embarked)
    5. Select feature columns
    6. Scale features with StandardScaler

    Returns X_train, X_test, y_train, y_test, scaler, feature_names
    """
    data = df.copy()

    # Step 1: Drop deck (>77% missing)
    data.drop(columns=["deck"], inplace=True, errors="ignore")

    # Step 2: Impute missing Age with median
    age_median = data["age"].median()
    data["age"] = data["age"].fillna(age_median)

    # Step 3: Impute missing Embarked with mode
    embarked_mode = data["embarked"].mode()[0]
    data["embarked"] = data["embarked"].fillna(embarked_mode)

    # Step 4: Encode categorical features
    data["sex"] = data["sex"].map({"male": 0, "female": 1})
    embarked_map = {"S": 0, "C": 1, "Q": 2}
    data["embarked"] = data["embarked"].map(embarked_map).fillna(0).astype(int)

    # Step 5: Select features and target
    feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    X = data[feature_cols]
    y = data["survived"]

    # Drop rows with remaining NaN
    valid_mask = X.notna().all(axis=1) & y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 6: Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


@st.cache_resource
def get_ml_model():
    """
    Train and cache the ensemble Voting Classifier on the Titanic dataset.
    Returns: (model, scaler, accuracy, report, cm)
    """
    df = load_titanic_data()
    X_train, X_test, y_train, y_test, scaler, feature_cols = preprocess_titanic(df)

    # Build ensemble with 4 base estimators
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    svc = SVC(probability=True, random_state=42)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("lr", lr), ("svc", svc)],
        voting="soft",
    )
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return ensemble, scaler, accuracy, report, cm


def predict_survival(model, scaler, pclass, sex, age, sibsp, parch, fare, embarked):
    """
    Predict survival probability for a single passenger.

    Parameters
    ----------
    model   : trained VotingClassifier
    scaler  : fitted StandardScaler
    pclass  : int  (1, 2, or 3)
    sex     : str  ('male' or 'female')
    age     : float
    sibsp   : int
    parch   : int
    fare    : float
    embarked: str  ('S', 'C', or 'Q')

    Returns
    -------
    prediction : int  (0 = not survived, 1 = survived)
    probability: list [prob_not_survived, prob_survived]
    """
    sex_enc = 1 if sex == "female" else 0
    embarked_enc = {"S": 0, "C": 1, "Q": 2}.get(embarked, 0)

    features = np.array([[pclass, sex_enc, age, sibsp, parch, fare, embarked_enc]])
    # Suppress benign sklearn feature-name warning (numpy array vs DataFrame)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0].tolist()

    return int(prediction), probability
