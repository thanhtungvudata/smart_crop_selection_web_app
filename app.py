import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scripts.preprocess import preprocess_input
from sklearn.metrics import accuracy_score

# Load trained model and label encoder
crop_model = joblib.load('models/crop_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Streamlit UI
st.title("🌱 Smart Crop Selection Web App")
st.markdown("Enter soil parameters to get the best crop recommendation.")

# Sidebar inputs
st.sidebar.header("Input Soil Parameters")
N = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, max_value=160.0, value=30.0)
P = st.sidebar.number_input("Phosphorus (P)", min_value=0.0, max_value=160.0, value=60.0)
K = st.sidebar.number_input("Potassium (K)", min_value=0.0, max_value=250.0, value=50.0)
pH = st.sidebar.number_input("pH Level", min_value=0.0, max_value=10.0, value=6.0)

# Convert inputs to DataFrame
input_data = pd.DataFrame([[N, P, K, pH]], columns=['N', 'P', 'K', 'ph'])

# Preprocess input
test_input = preprocess_input(input_data)

# Prediction
if st.sidebar.button("Predict Crop"):
    predicted_crop_index = crop_model.predict(test_input)[0]
    recommended_crop = label_encoder.inverse_transform([predicted_crop_index])[0]
    
    # Calculate accuracy on training data
    training_data = pd.read_csv('data/soil_measures.csv')
    X_train, y_train, _, _ = preprocess_input(training_data.drop(columns=['crop'])), training_data['crop'], None, None
    y_train_encoded = label_encoder.transform(y_train)
    y_train_pred = crop_model.predict(X_train)
    accuracy = accuracy_score(y_train_encoded, y_train_pred) * 100
    
    # Display results
    st.subheader("🌾 Recommended Crop:")
    st.write(f"**{recommended_crop}** with an accuracy of **{accuracy:.2f}%**")
