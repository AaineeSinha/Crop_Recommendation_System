import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("model/crop_model.pkl")

st.set_page_config(page_title="Crop Recommendation System")

st.title("🌾 Crop Recommendation System")
st.write("Enter soil and weather details to get the best crop recommendation.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0)
P = st.number_input("Phosphorus (P)", min_value=0)
K = st.number_input("Potassium (K)", min_value=0)
temperature = st.number_input("Temperature (°C)")
humidity = st.number_input("Humidity (%)")
ph = st.number_input("Soil pH")
rainfall = st.number_input("Rainfall (mm)")

if st.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    st.success(f"✅ Recommended Crop: **{prediction[0]}**")
