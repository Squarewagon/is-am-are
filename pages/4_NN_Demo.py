import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.nn_model import get_nn_model, predict_species, SPECIES_NAMES

st.set_page_config(page_title="NN Model Demo", page_icon="🔮", layout="wide")

st.title("🔮 Neural Network Model — Iris Species Demo")
st.markdown(
    "Enter flower measurements below to classify the Iris species using the trained MLP Neural Network."
)
st.markdown("---")

# Load model
with st.spinner("Loading model…"):
    mlp, scaler, le, accuracy, report, cm = get_nn_model()

st.success(f"✅ Model loaded — Test Accuracy: **{accuracy:.1%}**")

st.markdown("---")

# ── Example reference values ─────────────────────────────────────────────────
SPECIES_DEFAULTS = {
    "Iris setosa": (5.1, 3.5, 1.4, 0.2),
    "Iris versicolor": (5.9, 2.8, 4.3, 1.3),
    "Iris virginica": (6.5, 3.0, 5.5, 1.8),
}

st.subheader("Quick Fill (example values)")
quick_col1, quick_col2, quick_col3, _ = st.columns([1, 1, 1, 1])
preset = None
with quick_col1:
    if st.button("🌸 Setosa example", use_container_width=True):
        preset = "Iris setosa"
with quick_col2:
    if st.button("🌼 Versicolor example", use_container_width=True):
        preset = "Iris versicolor"
with quick_col3:
    if st.button("🌺 Virginica example", use_container_width=True):
        preset = "Iris virginica"

# Determine default slider values
if preset:
    sl_def, sw_def, pl_def, pw_def = SPECIES_DEFAULTS[preset]
else:
    sl_def, sw_def, pl_def, pw_def = 5.1, 3.5, 1.4, 0.2

st.markdown("---")

# ── Input Form ──────────────────────────────────────────────────────────────
st.subheader("Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider(
        "Sepal Length (cm)", min_value=4.0, max_value=8.0, value=sl_def, step=0.1
    )
    sepal_width = st.slider(
        "Sepal Width (cm)", min_value=2.0, max_value=5.0, value=sw_def, step=0.1
    )

with col2:
    petal_length = st.slider(
        "Petal Length (cm)", min_value=1.0, max_value=7.0, value=pl_def, step=0.1
    )
    petal_width = st.slider(
        "Petal Width (cm)", min_value=0.1, max_value=3.0, value=pw_def, step=0.1
    )

st.markdown("---")

# ── Prediction ──────────────────────────────────────────────────────────────
if st.button("🔮 Classify Species", type="primary", use_container_width=True):
    pred_idx, species_name, probability = predict_species(
        mlp, scaler, sepal_length, sepal_width, petal_length, petal_width
    )

    st.markdown("### Prediction Result")
    col_res, col_bar = st.columns([1, 2])

    species_emoji = {"setosa": "🌸", "versicolor": "🌼", "virginica": "🌺"}

    with col_res:
        emoji = species_emoji.get(species_name, "🌷")
        st.success(f"### {emoji} Iris *{species_name}*")

        st.markdown("**Confidence per class:**")
        for i, (sp, prob) in enumerate(zip(SPECIES_NAMES, probability)):
            e = species_emoji.get(sp, "🌷")
            st.progress(prob, text=f"{e} {sp}: {prob:.1%}")

        feature_summary = pd.DataFrame(
            {
                "Feature": [
                    "Sepal Length (cm)",
                    "Sepal Width (cm)",
                    "Petal Length (cm)",
                    "Petal Width (cm)",
                ],
                "Value": [sepal_length, sepal_width, petal_length, petal_width],
            }
        )
        st.table(feature_summary)

    with col_bar:
        fig = go.Figure(
            go.Bar(
                x=SPECIES_NAMES,
                y=[p * 100 for p in probability],
                marker_color=["#FF69B4", "#FFD700", "#9370DB"],
                text=[f"{p:.1%}" for p in probability],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Predicted Probability per Species (%)",
            yaxis_title="Probability (%)",
            xaxis_title="Species",
            yaxis_range=[0, 110],
            height=350,
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(
    """
    **How this works:**  
    Your flower measurements are scaled using the same `StandardScaler` fitted during training,  
    then passed through the 4-layer MLP (64→32→16 neurons). The final Softmax output gives  
    probabilities for each of the 3 Iris species.

    **Tip:** Use the *Quick Fill* buttons above to load typical values for each species,  
    then tweak individual measurements to see how the model responds.
    """
)
