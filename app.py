import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model pipeline
model = joblib.load('best_medical_cost_model.pkl')

st.set_page_config(page_title="Medical Cost Predictor", layout="centered")

st.title("🏥 Medical Cost Prediction App")
st.write("This app predicts the expected medical cost based on health and demographic factors.")

st.sidebar.header("User Input Features")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 65, 30)
    sex = st.sidebar.selectbox("Sex", ("male", "female"))
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 25.0)
    children = st.sidebar.slider("Children", 0, 5, 0)
    smoker = st.sidebar.selectbox("Smoker", ("yes", "no"))
    region = st.sidebar.selectbox("Region", ("northeast", "northwest", "southeast", "southwest"))
    
    data = {
        'Age': age,
        'Sex': sex,
        'BMI': bmi,
        'Children': children,
        'Smoker': smoker,
        'Region': region
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader("User Input Parameters")
st.write(input_df)

if st.button("Predict Cost"):
    prediction = model.predict(input_df)
    st.subheader("Predicted Medical Cost")
    st.success(f"${prediction[0]:,.2f}")