# is-am-are | Intelligent Systems ML Project

A Streamlit web application demonstrating **Ensemble Machine Learning** and **Neural Network** models for an intelligent systems course project.

## [Website](https://thepain.streamlit.app/)

## Datasets

| # | Dataset | Source | Task | Imperfections |
|---|---------|--------|------|---------------|
| 1 | **Titanic** | Seaborn built-in / Kaggle / OpenML | Binary classification (survival) | Missing Age (19.9%), missing Deck (77.2%), missing Embarked (0.2%) |
| 2 | **Iris** (modified) | Scikit-learn built-in (R.A. Fisher, 1936) | Multi-class classification (3 species) | ~10% missing values, 8 duplicates, 2 outliers |

## Models

| Model | Type | Base Estimators | Dataset |
|-------|------|-----------------|---------|
| **Ensemble Voting Classifier** | ML Ensemble | Random Forest + Gradient Boosting + Logistic Regression + SVC | Titanic |
| **MLP Neural Network** | Neural Network (64→32→16) | — (single model, Adam optimiser, ReLU) | Iris |