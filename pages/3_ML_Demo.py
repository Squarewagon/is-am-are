import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ml_model import get_ml_model, predict_survival

st.set_page_config(page_title="ML Model Demo", layout="wide")

st.title("Ensemble ML Model — Titanic Survival Demo")
st.markdown(
    "Enter passenger details below to predict the probability of survival using the trained Voting Classifier."
)
st.markdown("---")

# Load model
with st.spinner("Loading model…"):
    ensemble, scaler, accuracy, report, cm = get_ml_model()

st.success(f"Model loaded — Test Accuracy: **{accuracy:.1%}**")

st.markdown("---")

# ── Input Form ──────────────────────────────────────────────────────────────
st.subheader("Passenger Details")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        index=2,
        help="1st class = upper deck (most expensive), 3rd class = lower deck",
    )
    sex = st.radio("Sex", options=["male", "female"], index=0)
    age = st.slider("Age", min_value=1, max_value=80, value=28, step=1)

with col2:
    sibsp = st.number_input(
        "Siblings / Spouses Aboard",
        min_value=0,
        max_value=8,
        value=0,
        step=1,
    )
    parch = st.number_input(
        "Parents / Children Aboard",
        min_value=0,
        max_value=6,
        value=0,
        step=1,
    )

with col3:
    fare = st.number_input(
        "Ticket Fare (£)",
        min_value=0.0,
        max_value=600.0,
        value=15.0,
        step=1.0,
        format="%.2f",
    )
    embarked = st.selectbox(
        "Port of Embarkation",
        options=["S", "C", "Q"],
        format_func=lambda x: {"S": "S — Southampton", "C": "C — Cherbourg", "Q": "Q — Queenstown"}[x],
        index=0,
    )

st.markdown("---")

# ── Prediction ──────────────────────────────────────────────────────────────
if st.button("Predict Survival", type="primary", use_container_width=True):
    prediction, probability = predict_survival(
        ensemble, scaler, pclass, sex, age, sibsp, parch, fare, embarked
    )

    st.markdown("### Prediction Result")
    col_res, col_gauge = st.columns([1, 2])

    with col_res:
        if prediction == 1:
            st.success("### Survived 😭")
        else:
            st.error("### Did Not Survive 🤣")

        st.metric("Survival Probability", f"{probability[1]:.1%}")
        st.metric("Non-Survival Probability", f"{probability[0]:.1%}")

        feature_summary = pd.DataFrame(
            {
                "Feature": [
                    "Passenger Class",
                    "Sex",
                    "Age",
                    "Siblings/Spouses",
                    "Parents/Children",
                    "Fare (£)",
                    "Embarked",
                ],
                "Value": [
                    f"{pclass}",
                    sex,
                    f"{age}",
                    f"{sibsp}",
                    f"{parch}",
                    f"{fare:.2f}",
                    embarked,
                ],
            }
        )
        st.table(feature_summary)

    with col_gauge:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability[1] * 100,
                title={"text": "Survival Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "green" if prediction == 1 else "red"},
                    "steps": [
                        {"range": [0, 50], "color": "#ffcccc"},
                        {"range": [50, 100], "color": "#ccffcc"},
                    ],
                    "threshold": {
                        "line": {"color": "navy", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            )
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(
    """
    **How this works:**  
    Your input is preprocessed (sex/embarked encoded, features scaled) and passed through  
    the Voting Classifier. The 4 base models each output a survival probability; the final  
    probability is the average of those 4 predictions (soft voting).
    """
)
