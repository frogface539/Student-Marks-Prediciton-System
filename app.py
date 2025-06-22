import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# Load models and expected columns
with open("ada_boost_model.pkl", "rb") as f:
    ada_model = pickle.load(f)

with open("gradient_boost_model.pkl", "rb") as f:
    gbr_model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    expected_columns = pickle.load(f)

st.set_page_config(page_title="Student Score Predictor", layout="centered")

st.title("🎓 Student Average Score Predictor")

st.markdown("Get performance predictions using **AdaBoost** and **Gradient Boosting** techniques.")

st.subheader("📋 Enter Student Details")

# Input features
gender = st.selectbox("Gender", ["male", "female"])
race_ethnicity = st.selectbox("Race/Ethnicity", ['group A', 'group B', 'group C', 'group D', 'group E'])
parent_education = st.selectbox("Parental Level of Education", [
    "high school", "some high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])
math_score = st.slider("📊 Math Score", 0, 100, 75)
reading_score = st.slider("📖 Reading Score", 0, 100, 75)
writing_score = st.slider("✍️ Writing Score", 0, 100, 75)

# Predict button
if st.button("🔍 Predict Average Score"):

    input_dict = {
        "gender": gender,
        "race/ethnicity": race_ethnicity,
        "parental level of education": parent_education,
        "lunch": lunch,
        "test preparation course": test_prep,
        "math_score": math_score,
        "reading_score": reading_score,
        "writing_score": writing_score
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)

    for col in expected_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[expected_columns]

    ada_pred = ada_model.predict(input_encoded)[0]
    gbr_pred = gbr_model.predict(input_encoded)[0]

    st.header("🎯 Predicted Average Scores")

    # Gauge function
    def draw_gauge(prediction, title, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': "#ff6666"},       # Fail
                    {'range': [40, 70], 'color': "#ffe066"},      # Average
                    {'range': [70, 100], 'color': "#66ff66"}      # Pass
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction
                }
            }
        ))
        return fig

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("🔹 **AdaBoost Regressor**")
        st.plotly_chart(draw_gauge(ada_pred, "AdaBoost", "#000000"), use_container_width=True)

    with col2:
        st.markdown("🔸 **Gradient Boosting Regressor**")
        st.plotly_chart(draw_gauge(gbr_pred, "Gradient Boost", "#000000"), use_container_width=True)

    st.success(f"✅ Based on your inputs, the predicted average score ranges between **{min(ada_pred, gbr_pred):.2f}** and **{max(ada_pred, gbr_pred):.2f}**.")
