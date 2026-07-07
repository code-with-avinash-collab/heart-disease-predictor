import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load("heart_disease_model.pkl")

# Title
st.title("❤️ Heart Disease Prediction")

st.write("Enter Patient Details")

# Inputs
age = st.number_input("Age", min_value=1, max_value=120, value=45)

# Gender
sex = st.selectbox("Gender", ["Male", "Female"])

if sex == "Male":
    sex = 1
else:
    sex = 0

# Chest Pain Type
cp = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ]
)

if cp == "Typical Angina":
    cp = 0
elif cp == "Atypical Angina":
    cp = 1
elif cp == "Non-anginal Pain":
    cp = 2
else:
    cp = 3

trestbps = st.number_input("Resting Blood Pressure", value=120)

chol = st.number_input("Cholesterol", value=200)

# Fasting Blood Sugar
fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["Yes", "No"]
)

if fbs == "Yes":
    fbs = 1
else:
    fbs = 0

# Resting ECG
restecg = st.selectbox(
    "Resting ECG",
    [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ]
)

if restecg == "Normal":
    restecg = 0
elif restecg == "ST-T Wave Abnormality":
    restecg = 1
else:
    restecg = 2

thalach = st.number_input("Maximum Heart Rate", value=150)

# Exercise Induced Angina
exang = st.selectbox(
    "Exercise Induced Angina",
    ["Yes", "No"]
)

if exang == "Yes":
    exang = 1
else:
    exang = 0

oldpeak = st.number_input("Oldpeak", value=1.0)

# Slope
slope = st.selectbox(
    "Slope of ST Segment",
    [
        "Upsloping",
        "Flat",
        "Downsloping"
    ]
)

if slope == "Upsloping":
    slope = 0
elif slope == "Flat":
    slope = 1
else:
    slope = 2

# Major Vessels
ca = st.selectbox(
    "Number of Major Blood Vessels",
    [0, 1, 2, 3, 4]
)

# Thal
thal = st.selectbox(
    "Thal",
    [
        "Normal",
        "Fixed Defect",
        "Reversible Defect",
        "Unknown"
    ]
)

if thal == "Normal":
    thal = 0
elif thal == "Fixed Defect":
    thal = 1
elif thal == "Reversible Defect":
    thal = 2
else:
    thal = 3

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
        st.write(f"Probability : {probability[0][1]*100:.2f}%")
    else:
        st.success("✅ No Heart Disease Detected")
        st.write(f"Probability : {probability[0][0]*100:.2f}%")
