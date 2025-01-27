import numpy as np
import pickle
import streamlit as st

# Load the trained model
model_path = "parkinsons_model.sav"
loaded_model = pickle.load(open("C:/Final Year Project/parkinson Disease/parkinsons_model.sav", 'rb'))

# Function for Parkinson's disease prediction
def parkinsons_prediction(input_data):
    """
    Predict if a person has Parkinson's disease based on input data.

    Parameters:
        input_data (list): A list of numerical values corresponding to input features.

    Returns:
        str: Prediction result (Parkinson's disease or not).
    """
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape for a single prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 1:
        return "The person is having Parkinson's disease."
    else:
        return "The person does not have Parkinson's disease."

# Streamlit Web App
def main():
    # Set page title
    st.title("Parkinson's Disease Prediction System")

    # Input fields for user data
    st.write("Enter the following features for prediction:")
    
    # Collect features as input
    mdvp_fo = st.text_input("MDVP:Fo (Hz)")
    mdvp_fhi = st.text_input("MDVP:Fhi (Hz)")
    mdvp_flo = st.text_input("MDVP:Flo (Hz)")
    mdvp_jitter_percent = st.text_input("MDVP:Jitter(%)")
    mdvp_jitter_abs = st.text_input("MDVP:Jitter(Abs)")
    mdvp_rap = st.text_input("MDVP:RAP")
    mdvp_ppq = st.text_input("MDVP:PPQ")
    jitter_ddp = st.text_input("Jitter:DDP")
    mdvp_shimmer = st.text_input("MDVP:Shimmer")
    mdvp_shimmer_db = st.text_input("MDVP:Shimmer(dB)")
    shimmer_apq3 = st.text_input("Shimmer:APQ3")
    shimmer_apq5 = st.text_input("Shimmer:APQ5")
    shimmer_dda = st.text_input("Shimmer:DDA")
    nhr = st.text_input("NHR")
    hnr = st.text_input("HNR")

    # Prediction result
    diagnosis = ""

    # Button for prediction
    if st.button("Get Parkinson's Test Result"):
        try:
            # Collect user input
            user_input = [
                float(mdvp_fo), float(mdvp_fhi), float(mdvp_flo), float(mdvp_jitter_percent),
                float(mdvp_jitter_abs), float(mdvp_rap), float(mdvp_ppq), float(jitter_ddp),
                float(mdvp_shimmer), float(mdvp_shimmer_db), float(shimmer_apq3),
                float(shimmer_apq5), float(shimmer_dda), float(nhr), float(hnr)
            ]

            # Make prediction
            diagnosis = parkinsons_prediction(user_input)
        except ValueError:
            diagnosis = "Please enter valid numerical values for all fields."

    # Display the prediction
    st.success(diagnosis)

if __name__ == "__main__":
    main()
