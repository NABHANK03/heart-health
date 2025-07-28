import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# --- Load Model and Feature Names ---
# Use st.cache_resource to load these only once
@st.cache_resource
def load_resources():
    """Loads the trained model and feature names."""
    with open('heart_disease_model.pkl', 'rb') as f_model:
        model = pickle.load(f_model)
    with open('feature_names.pkl', 'rb') as f_features:
        feature_names = pickle.load(f_features)
    return model, feature_names

model, feature_names = load_resources()

# --- App Title and Description ---
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Risk Predictor")
st.write(
    "This app uses a Machine Learning model to predict the risk of heart disease "
    "based on 13 clinical features. The model was trained on the UCI Heart Disease dataset "
    "and achieved a high level of accuracy."
)
st.markdown("---")

# --- Sidebar for User Input ---
st.sidebar.header("Enter Patient Information")

# Function to create input fields
def user_input_features():
    age = st.sidebar.slider("Age", 29, 77, 54)
    sex = st.sidebar.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}[x])
    trestbps = st.sidebar.slider("Resting Blood Pressure (trestbps)", 94, 200, 130)
    chol = st.sidebar.slider("Serum Cholestoral (chol) in mg/dl", 126, 564, 240)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1], format_func=lambda x: "False" if x == 0 else "True")
    restecg = st.sidebar.selectbox("Resting Electrocardiographic Results (restecg)", [0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T wave abnormality", 2: "Probable or definite left ventricular hypertrophy"}[x])
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved (thalach)", 71, 202, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.sidebar.slider("ST depression induced by exercise (oldpeak)", 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox("Slope of the peak exercise ST segment (slope)", [0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
    ca = st.sidebar.selectbox("Number of major vessels colored by flourosopy (ca)", [0, 1, 2, 3, 4])
    thal = st.sidebar.selectbox("Thalassemia (thal)", [0, 1, 2, 3], format_func=lambda x: {0: "Unknown", 1: "Normal", 2: "Fixed defect", 3: "Reversable defect"}[x])

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs,
        'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak,
        'slope': slope, 'ca': ca, 'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user input
st.subheader("Patient Input Features")
st.write(input_df)

# --- Prediction and Interpretation ---
if st.button("Predict Heart Disease Risk"):
    # Reorder columns to match model's training order
    input_df = input_df[feature_names]

    # Prediction
    prediction_proba = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    st.header("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Score")
        risk_percentage = prediction_proba * 100
        if prediction == 1:
            st.error(f"High Risk ({risk_percentage:.2f}%)")
            st.write("The model predicts this patient has heart disease.")
        else:
            st.success(f"Low Risk ({risk_percentage:.2f}%)")
            st.write("The model predicts this patient does not have heart disease.")

    # SHAP Explanation
    with st.expander("See Prediction Explanation (Feature Importance)"):
        st.write(
            "The chart below shows which features pushed the model's prediction "
            "towards 'High Risk' (red bars) or 'Low Risk' (blue bars) for this specific patient."
        )
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_values[1][0], max_display=14, show=False)
        st.pyplot(fig)

# --- Footer ---
st.markdown("---")
st.write("**Disclaimer:** This is an educational tool and not a substitute for professional medical advice.")
