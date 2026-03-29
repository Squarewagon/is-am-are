import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ml_model import load_titanic_data, preprocess_titanic, get_ml_model

st.set_page_config(page_title="ML Model Info", layout="wide")

st.title("Ensemble Machine Learning Model")
st.markdown("### Titanic Survival Prediction — Model Description")
st.markdown("---")

# ── 1. Dataset Description ─────────────────────────────────────────────────
st.header("1. Dataset Description")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        **Dataset:** Titanic Passenger Survival  
        **Source:** Seaborn built-in dataset (originally from Kaggle / OpenML — publicly available)  
        **Task:** Binary Classification — predict whether a passenger survived (1) or not (0)  
        **Size:** 891 rows × 9 columns

        | Feature | Description | Type |
        |---------|-------------|------|
        | `survived` | Survival (0 = No, 1 = Yes) | Target |
        | `pclass` | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) | Ordinal |
        | `sex` | Gender of the passenger | Categorical |
        | `age` | Age in years | Numeric |
        | `sibsp` | # of siblings/spouses aboard | Numeric |
        | `parch` | # of parents/children aboard | Numeric |
        | `fare` | Passenger fare (£) | Numeric |
        | `embarked` | Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton) | Categorical |
        | `deck` | Deck letter (A–G) | Categorical |
        """
    )

with col2:
    df_raw = load_titanic_data()
    st.markdown("**Missing Values in Raw Data**")
    missing = df_raw.isnull().sum().reset_index()
    missing.columns = ["Feature", "Missing Count"]
    missing["Missing %"] = (missing["Missing Count"] / len(df_raw) * 100).round(1)
    st.dataframe(missing, hide_index=True)

with st.expander("Show raw data sample (first 10 rows)"):
    st.dataframe(df_raw.head(10))

st.markdown("---")

# ── 2. Data Preparation ─────────────────────────────────────────────────────
st.header("2. Data Preparation Steps")
st.markdown(
    """
    The raw dataset contained several imperfections that required preprocessing before model training:

    **Step 1 — Drop high-missing column (`deck`)**  
    The `deck` column had 77% missing values, making imputation unreliable.  
    → **Action:** Drop the column entirely.

    **Step 2 — Impute missing `age` values**  
    19.9% of age values were missing (177 rows).  
    → **Action:** Replace `NaN` with the **median age** (28.0 years) — robust to skew.

    **Step 3 — Impute missing `embarked` values**  
    Only 2 rows had missing embarked ports.  
    → **Action:** Replace `NaN` with the **mode** ('S' = Southampton, most common).

    **Step 4 — Encode categorical features**  
    Machine learning models require numeric inputs.  
    → `sex`: `male` → 0, `female` → 1  
    → `embarked`: `S` → 0, `C` → 1, `Q` → 2

    **Step 5 — Feature selection**  
    Selected 7 predictive features: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`.

    **Step 6 — Train/Test split and feature scaling**  
    → 80% training / 20% test split (random_state=42)  
    → Features normalized using **StandardScaler** (zero mean, unit variance)
    """
)

# ── 3. Algorithm Theory ─────────────────────────────────────────────────────
st.header("3. Algorithm Theory — Ensemble Voting Classifier")
st.markdown(
    """
    An **ensemble model** combines multiple base learners to produce a stronger predictor.  
    The **Voting Classifier** aggregates predictions from all base models:

    - **Soft Voting:** Each classifier outputs class probabilities; the final prediction is the  
      class with the highest *average* probability across all models.

    $$\\hat{y} = \\arg\\max_k \\frac{1}{N} \\sum_{i=1}^{N} P_i(y = k \\mid x)$$

    This ensemble is composed of **4 base estimators**:

    | # | Model | Key Characteristics |
    |---|-------|---------------------|
    | 1 | **Random Forest** | Bagging of 100 decision trees; reduces variance via feature & row sampling |
    | 2 | **Gradient Boosting** | Additive boosting of 100 shallow trees; reduces bias sequentially |
    | 3 | **Logistic Regression** | Linear model estimating log-odds; works well when features are linearly separable |
    | 4 | **SVC (SVM)** | Maximises the decision boundary margin; effective in high-dimensional spaces |

    **Why soft voting?** Averaging probabilities uses more information than hard majority voting,  
    leading to better-calibrated predictions, especially when base models are diverse.
    """
)

# ── 4. Model Development Steps ──────────────────────────────────────────────
st.header("4. Model Development Steps")
st.markdown(
    """
    1. Load raw Titanic CSV  
    2. Apply data preparation pipeline (Steps 1–6 above)  
    3. Instantiate four base estimators with tuned hyperparameters  
    4. Wrap in `sklearn.ensemble.VotingClassifier` with `voting='soft'`  
    5. Fit ensemble on 80% training data  
    6. Evaluate on 20% held-out test set  
    7. Cache trained model with `@st.cache_resource` for fast inference in the demo
    """
)

# ── 5. Model Performance ────────────────────────────────────────────────────
st.header("5. Model Performance")
with st.spinner("Training ensemble model (first load only)…"):
    ensemble, scaler, accuracy, report, cm = get_ml_model()

st.metric("Test Accuracy", f"{accuracy:.1%}")

col_rep, col_cm = st.columns(2)

with col_rep:
    st.markdown("**Classification Report**")
    rep_df = pd.DataFrame(report).T.round(3)
    st.dataframe(rep_df)

with col_cm:
    st.markdown("**Confusion Matrix**")
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Survived", "Survived"])
    ax.set_yticklabels(["Not Survived", "Survived"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── 6. References ────────────────────────────────────────────────────────────
st.header("6. References")
st.markdown(
    """
    1. Pedregosa, F. et al. (2011). **Scikit-learn: Machine learning in Python.**  
       *Journal of Machine Learning Research*, 12, 2825–2830.  
       https://scikit-learn.org/
    2. Titanic dataset — Kaggle / OpenML.  
       https://www.openml.org/search?type=data&sort=runs&id=40945
    3. Breiman, L. (2001). **Random Forests.** *Machine Learning*, 45(1), 5–32.
    4. Friedman, J. H. (2001). **Greedy function approximation: A gradient boosting machine.**  
       *Annals of Statistics*, 29(5), 1189–1232.
    5. Cortes, C. & Vapnik, V. (1995). **Support-vector networks.** *Machine Learning*, 20(3), 273–297.
    """
)
