import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
diabetes_data = pd.read_csv('diabetes.csv')

# Prepare the data
X = diabetes_data.drop(columns='Outcome', axis=1)
Y = diabetes_data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Train models
model_svc = SVC(kernel='linear')
model_lr = LogisticRegression()

model_svc.fit(X_train, Y_train)
model_lr.fit(X_train, Y_train)

# Streamlit UI
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.write("Enter the details below to predict if a person is diabetic:")

# User input
preg = st.number_input('Pregnancies', min_value=0, value=1)
glucose = st.number_input('Glucose', min_value=0, value=120)
bp = st.number_input('Blood Pressure', min_value=0, value=70)
skin = st.number_input('Skin Thickness', min_value=0, value=20)
insulin = st.number_input('Insulin', min_value=0, value=79)
bmi = st.number_input('BMI', min_value=0.0, format="%.1f", value=25.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, format="%.3f", value=0.5)
age = st.number_input('Age', min_value=1, value=33)

input_data = np.array([preg, glucose, bp, skin, insulin, bmi, dpf, age]).reshape(1, -1)
scaled_input = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    pred_svc = model_svc.predict(scaled_input)[0]
    pred_lr = model_lr.predict(scaled_input)[0]

    st.subheader("Prediction Results")
    st.write(f"**SVC Model:** {'Diabetic' if pred_svc else 'Not Diabetic'}")
    st.write(f"**Logistic Regression Model:** {'Diabetic' if pred_lr else 'Not Diabetic'}")
