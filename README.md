# is-am-are — Intelligent Systems ML Project

A Streamlit web application demonstrating **Ensemble Machine Learning** and **Neural Network** models for an intelligent systems course project.

## 📊 Datasets

| # | Dataset | Source | Task | Imperfections |
|---|---------|--------|------|---------------|
| 1 | **Titanic** | Seaborn built-in / Kaggle / OpenML | Binary classification (survival) | Missing Age (19.9%), missing Deck (77.2%), missing Embarked (0.2%) |
| 2 | **Iris** (modified) | Scikit-learn built-in (R.A. Fisher, 1936) | Multi-class classification (3 species) | ~10% missing values, 8 duplicates, 2 outliers |

## 🤖 Models

| Model | Type | Base Estimators | Dataset |
|-------|------|-----------------|---------|
| **Ensemble Voting Classifier** | ML Ensemble | Random Forest + Gradient Boosting + Logistic Regression + SVC | Titanic |
| **MLP Neural Network** | Neural Network (64→32→16) | — (single model, Adam optimiser, ReLU) | Iris |

## 🚀 Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📑 Pages

1. **Home** — Project overview, dataset summaries, model summaries
2. **ML Model Info** — Theory, data prep steps, performance metrics for the Ensemble model
3. **NN Model Info** — Theory, MLP architecture, data prep steps, performance metrics
4. **ML Model Demo** — Interactive Titanic survival predictor
5. **NN Model Demo** — Interactive Iris species classifier

## 🌐 Deployment

The app can be deployed to [Streamlit Community Cloud](https://streamlit.io/cloud) by pointing it at this repository and setting the main file to `app.py`.
