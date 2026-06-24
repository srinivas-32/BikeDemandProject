
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("Bike Rental Prediction - Azure Designer Equivalent")

cols = [
'instant','season','yr','mnth','hr','holiday','weekday',
'workingday','weathersit','temp','atemp','hum',
'windspeed','casual','registered'
]

values = {}
for c in cols:
    values[c] = st.number_input(c, value=0.0)

if st.button("Predict"):
    df = pd.DataFrame([values])
    pred = model.predict(df)[0]
    st.success(f"Predicted Bike Count: {pred:.2f}")
