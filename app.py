# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load ANN model
model = load_model("model.h5")

# Load scaler and encoders
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder_gender.pkl", "rb") as f:
    le_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    ohe_geo = pickle.load(f)

st.title("Customer Churn Prediction (ANN)")

st.write("Enter customer details to predict whether they will churn:")

# User Inputs
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
balance = st.number_input("Balance", min_value=0.0, value=1000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=10, value=1)
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# Preprocess inputs
gender_encoded = le_gender.transform([gender])[0]
has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
is_active_member_encoded = 1 if is_active_member == "Yes" else 0

# One-hot encode geography
geo_array = ohe_geo.transform([[geography]]).toarray()

# Combine all features
input_data = np.array([[credit_score, gender_encoded, age, tenure, balance, 
                        num_of_products, has_cr_card_encoded, is_active_member_encoded, estimated_salary]])

input_data = np.concatenate([input_data, geo_array], axis=1)

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Churn"):
    pred = model.predict(input_scaled)[0][0]
    result = "Churn" if pred > 0.5 else "Not Churn"
    st.success(f"The customer is likely to: {result}")
