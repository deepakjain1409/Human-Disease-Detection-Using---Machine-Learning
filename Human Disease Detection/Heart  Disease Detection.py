import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the trained model
model_path = "heart_disease_model.sav"
loaded_model = pickle.load(open('C:/Final Year Project/Heart disease detection/heart_disease_model.sav', 'rb'))

# Function for heart disease prediction
def heart_disease_prediction(input_data):
    """
    Predict if a person has heart disease based on input data.

    Parameters:
        input_data (list): A list of 13 numerical values corresponding to input features.

    Returns:
        str: Prediction result (Heart disease or not).
    """
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape for a single prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 1:
        return "The person is having heart disease"
    else:
        return "The person does not have heart disease"

# Streamlit Web App
def main():
    # Set page title
    st.title("Heart Disease Prediction System")

    # Input fields for user data
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Age")
        trestbps = st.text_input("Resting Blood Pressure")
        chol = st.text_input("Serum Cholestoral (mg/dl)")
        thalach = st.text_input("Maximum Heart Rate Achieved")
        oldpeak = st.text_input("ST Depression Induced by Exercise")

    with col2:
        sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
        exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
        slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])

    with col3:
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        restecg = st.selectbox("Resting Electrocardiographic Results (0-2)", [0, 1, 2])
        ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", [0, 1, 2])

    # Prediction result
    diagnosis = ""

    # Button for prediction
    if st.button("Get Heart Disease Test Result"):
        try:
            # Collect user input
            user_input = [
                float(age), float(sex), float(cp), float(trestbps), float(chol),
                float(fbs), float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca), float(thal)
            ]

            # Make prediction
            diagnosis = heart_disease_prediction(user_input)
        except ValueError:
            diagnosis = "Please enter valid numerical values for all fields."

    # Display the prediction
    st.success(diagnosis)

if __name__ == "__main__":
    main()
