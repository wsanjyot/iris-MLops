from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "model.pkl"

# Must match the column names/order used in src/feature_engineering.py
FEATURES = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]

st.set_page_config(page_title="Iris Flower Prediction", page_icon="🌸", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def pretty(label: str) -> str:
    """'Iris-setosa' -> 'Setosa'"""
    return str(label).replace("Iris-", "").title()


st.title("🌸 Iris Flower Prediction App")

model = load_model()
if model is None:
    st.error(
        f"Model not found at `{MODEL_PATH}`. "
        "Run `dvc repro` (or `dvc pull`) to generate `models/model.pkl`."
    )
    st.stop()

st.write("Enter flower measurements below:")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=15.0, value=5.1, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=15.0, value=1.4, step=0.1)
with col2:
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=15.0, value=3.5, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=15.0, value=0.2, step=0.1)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]], columns=FEATURES
    )

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Flower: {pretty(prediction)}")

    # Class probabilities, only if the champion model supports them
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0]
        proba_df = pd.DataFrame(
            {"Probability": proba}, index=[pretty(c) for c in model.classes_]
        )
        st.bar_chart(proba_df)
