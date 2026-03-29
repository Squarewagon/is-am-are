import streamlit as st

st.set_page_config(
    page_title="IS-AM-ARE | ML & Neural Network Project",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Machine Learning & Neural Network Web Application")
st.markdown("### Intelligent Systems Project — IS-AM-ARE")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        ## 📊 Datasets Used

        ### 1. Titanic Survival Dataset
        - **Source:** Built-in Seaborn / open-source (Kaggle / OpenML)
        - **Type:** Structured (tabular)
        - **Task:** Binary classification (survived / not survived)
        - **Size:** 891 rows × 9 columns
        - **Imperfections:** 177 missing `Age` values, 688 missing `Deck` values, 2 missing `Embarked` values
        - **Features:** Passenger class, Sex, Age, Siblings/Spouses aboard, Parents/Children aboard, Fare, Port of embarkation

        ### 2. Iris Dataset (with Artificial Imperfections)
        - **Source:** Scikit-learn built-in dataset (originally by R.A. Fisher, 1936)
        - **Type:** Structured (tabular)
        - **Task:** Multi-class classification (3 flower species)
        - **Size:** 158 rows × 6 columns (original 150 + 8 duplicates added)
        - **Imperfections:** ~10% missing `sepal_length` & `petal_width`, 8 duplicate rows, 2 outlier values
        - **Features:** Sepal length, Sepal width, Petal length, Petal width
        """
    )

with col2:
    st.markdown(
        """
        ## 🤖 Models Developed

        ### Model 1 — Ensemble ML (Voting Classifier)
        Composed of **4 base estimators**:
        1. 🌲 **Random Forest** — Bagging ensemble of decision trees
        2. 📈 **Gradient Boosting** — Sequential boosting ensemble
        3. 📐 **Logistic Regression** — Linear probabilistic model
        4. 🔵 **Support Vector Classifier (SVC)** — Margin-based classifier

        Combined using **Soft Voting** (average predicted probabilities).  
        Trained on the **Titanic dataset**.

        ---

        ### Model 2 — Neural Network (MLP)
        **Multi-Layer Perceptron** with custom-designed architecture:
        - Input Layer: 4 features
        - Hidden Layer 1: **64 neurons** (ReLU)
        - Hidden Layer 2: **32 neurons** (ReLU)
        - Hidden Layer 3: **16 neurons** (ReLU)
        - Output Layer: **3 classes** (Softmax via cross-entropy)
        - Optimizer: **Adam**, with early stopping

        Trained on the **Iris dataset** (preprocessed).
        """
    )

st.markdown("---")

st.markdown(
    """
    ## 📑 Navigation

    Use the **sidebar** to navigate between pages:

    | Page | Description |
    |------|-------------|
    | 🧾 **ML Model Info** | Algorithm theory, data preparation steps, and references for the Ensemble ML model |
    | 🧾 **NN Model Info** | Algorithm theory, data preparation steps, and references for the Neural Network |
    | 🎯 **ML Model Demo** | Interactive demo — enter passenger details to predict Titanic survival |
    | 🔮 **NN Model Demo** | Interactive demo — enter flower measurements to predict Iris species |
    """
)

st.markdown("---")
st.caption("Intelligent Systems Project • Squarewagon/is-am-are")
