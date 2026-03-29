import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.nn_model import load_iris_data, preprocess_iris, get_nn_model, SPECIES_NAMES

st.set_page_config(page_title="NN Model Info", page_icon="🧠", layout="wide")

st.title("🧠 Neural Network Model")
st.markdown("### Iris Species Classification — Model Description")
st.markdown("---")

# ── 1. Dataset Description ─────────────────────────────────────────────────
st.header("1. Dataset Description")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        **Dataset:** Iris Flower Measurements (with Artificial Imperfections)  
        **Source:** Originally by R.A. Fisher (1936); reproduced in Scikit-learn built-in datasets.  
        Additional noise and duplicates were introduced for demonstration purposes.  
        **Task:** Multi-class Classification — identify the species of an Iris flower (3 classes)  
        **Size (raw):** 158 rows × 6 columns (150 original + 8 duplicates)

        | Feature | Description | Unit | Type |
        |---------|-------------|------|------|
        | `sepal_length` | Length of the sepal | cm | Numeric |
        | `sepal_width`  | Width of the sepal  | cm | Numeric |
        | `petal_length` | Length of the petal | cm | Numeric |
        | `petal_width`  | Width of the petal  | cm | Numeric |
        | `species` | Flower species (0=setosa, 1=versicolor, 2=virginica) | — | Target |

        **Introduced Imperfections:**
        - ~10% missing values in `sepal_length` and `petal_width`
        - 8 duplicate rows
        - 1 outlier in `sepal_length` (value = 15.0 cm — clearly above natural range ≤ 7.9 cm)
        - 1 outlier in `petal_width` (value = −1.0 cm — physically impossible)
        """
    )

with col2:
    df_raw = load_iris_data()
    st.markdown("**Missing Values in Raw Data**")
    missing = df_raw.isnull().sum().reset_index()
    missing.columns = ["Feature", "Missing Count"]
    missing["Missing %"] = (missing["Missing Count"] / len(df_raw) * 100).round(1)
    st.dataframe(missing, hide_index=True)

    st.markdown(f"**Duplicate rows:** {df_raw.duplicated().sum()}")

with st.expander("Show raw data sample (first 10 rows)"):
    st.dataframe(df_raw.head(10))

st.markdown("---")

# ── 2. Data Preparation ─────────────────────────────────────────────────────
st.header("2. Data Preparation Steps")
st.markdown(
    """
    **Step 1 — Remove duplicate rows**  
    8 duplicate rows were detected and removed to prevent data leakage.  
    → Remaining rows: 150

    **Step 2 — Remove outliers**  
    Values outside the plausible biological range were detected:  
    - `sepal_length` = 15.0 cm (max known value ≈ 7.9 cm) → removed  
    - `petal_width` = −1.0 cm (negative length is impossible) → removed

    **Step 3 — Impute remaining missing values**  
    After outlier removal, remaining NaN values (~10%) were imputed with the **column median** —  
    a robust statistic that is not sensitive to the outliers already removed.

    **Step 4 — Label encoding**  
    Target labels were encoded as integers: `setosa`=0, `versicolor`=1, `virginica`=2  
    using `sklearn.preprocessing.LabelEncoder`.

    **Step 5 — Train/Test split**  
    → 80% training / 20% test, stratified to preserve class balance, `random_state=42`.

    **Step 6 — Feature scaling**  
    Features scaled with **StandardScaler** (zero mean, unit variance) to ensure faster  
    and more stable gradient descent during MLP training.
    """
)

st.markdown("---")

# ── 3. Algorithm Theory ─────────────────────────────────────────────────────
st.header("3. Algorithm Theory — Multi-Layer Perceptron (MLP)")
st.markdown(
    """
    A **Multi-Layer Perceptron** is a fully connected feedforward neural network.  
    Each neuron computes a **weighted sum** of its inputs, adds a bias, and applies a  
    non-linear **activation function**:

    $$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \\quad a^{(l)} = f(z^{(l)})$$

    **Activation Functions:**
    - **Hidden layers:** ReLU — $f(x) = \\max(0, x)$ — avoids vanishing gradients, fast to compute
    - **Output layer:** Softmax — converts raw scores to probabilities for multi-class problems:

    $$P(y = k \\mid x) = \\frac{e^{z_k}}{\\sum_{j} e^{z_j}}$$

    **Training:**
    - Loss: **Cross-entropy** $\\mathcal{L} = -\\sum_k y_k \\log(\\hat{y}_k)$
    - Optimiser: **Adam** (adaptive learning rate, momentum)
    - Regularisation: **Early stopping** on a 10% validation split to prevent overfitting

    **Network Architecture (custom-designed):**
    """
)

# Architecture diagram (ASCII / text based)
arch_data = {
    "Layer": ["Input", "Hidden 1", "Hidden 2", "Hidden 3", "Output"],
    "Neurons": [4, 64, 32, 16, 3],
    "Activation": ["—", "ReLU", "ReLU", "ReLU", "Softmax"],
    "Description": [
        "4 flower measurements",
        "Feature extraction (wide)",
        "Feature compression",
        "Fine-grained features",
        "3 species probabilities",
    ],
}
st.dataframe(pd.DataFrame(arch_data), hide_index=True)

st.markdown("---")

# ── 4. Model Development Steps ──────────────────────────────────────────────
st.header("4. Model Development Steps")
st.markdown(
    """
    1. Load raw Iris CSV (with introduced imperfections)
    2. Apply data preparation pipeline (Steps 1–6 above)
    3. Instantiate `sklearn.neural_network.MLPClassifier` with architecture `(64, 32, 16)`
    4. Set `activation='relu'`, `solver='adam'`, `n_iter_no_change=20`, `max_iter=1000`
    5. Fit model on 80% training data
    6. Evaluate on 20% held-out test set
    7. Cache trained model with `@st.cache_resource` for fast inference
    """
)

# ── 5. Model Performance ────────────────────────────────────────────────────
st.header("5. Model Performance")
with st.spinner("Training neural network (first load only)…"):
    mlp, scaler, le, accuracy, report, cm = get_nn_model()

st.metric("Test Accuracy", f"{accuracy:.1%}")

# Training loss curve
col_rep, col_cm = st.columns(2)

with col_rep:
    st.markdown("**Classification Report**")
    rep_df = pd.DataFrame(report).T.round(3)
    st.dataframe(rep_df)

with col_cm:
    st.markdown("**Confusion Matrix**")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(SPECIES_NAMES, rotation=15)
    ax.set_yticklabels(SPECIES_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Loss curve
st.markdown("**Training Loss Curve**")
fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.plot(mlp.loss_curve_, label="Training Loss", color="steelblue")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Loss")
ax2.set_title("MLP Training Loss")
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.markdown("---")

# ── 6. References ────────────────────────────────────────────────────────────
st.header("6. References")
st.markdown(
    """
    1. Fisher, R. A. (1936). **The Use of Multiple Measurements in Taxonomic Problems.**  
       *Annals of Eugenics*, 7(2), 179–188.
    2. Pedregosa, F. et al. (2011). **Scikit-learn: Machine learning in Python.**  
       *Journal of Machine Learning Research*, 12, 2825–2830.  
       https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    3. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986).  
       **Learning representations by back-propagating errors.** *Nature*, 323, 533–536.
    4. Kingma, D. P. & Ba, J. (2015). **Adam: A Method for Stochastic Optimization.**  
       *ICLR 2015.* https://arxiv.org/abs/1412.6980
    5. Nair, V. & Hinton, G. E. (2010). **Rectified Linear Units Improve Restricted Boltzmann Machines.**  
       *ICML 2010.*
    """
)
