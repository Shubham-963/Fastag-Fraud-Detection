import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load the trained model
model = joblib.load('model.pkl')

# Load the CSV data
data = pd.read_csv("/content/FastagFraudDetection.csv")

# Function to preprocess input data
def preprocess_input(input_data):
    # Preprocessing steps specific to your data format
    # Here, you need to convert the input data into a format suitable for prediction
    # You can parse the input_data and prepare it accordingly
    # Example: convert categorical variables to numerical using LabelEncoder
    return input_data

# Function to make predictions using the trained model
def make_prediction(model, input_data):
    # Preprocess input data
    preprocessed_input = preprocess_input(input_data)
    # Make prediction
    prediction = model.predict(preprocessed_input)
    return prediction

# Main function to take user input and make predictions
def main():
    # Set page title
    st.title('Fastag Fraud Detection')

    # Get user input
    transaction_amount = st.number_input("Enter Transaction Amount:", min_value=0.0, value=0.0)
    amount_paid = st.number_input("Enter Amount Paid:", min_value=0.0, value=0.0)
    vehicle_speed = st.number_input("Enter Vehicle Speed:", min_value=0, value=0)

    # Make prediction
    if st.button("Predict"):
        user_input = np.array([[transaction_amount, amount_paid, vehicle_speed]])
        prediction = make_prediction(model, user_input)
        if prediction[0] == 1:
            st.write("Prediction: Scam")
        else:
            st.write("Prediction: Not a scam")

# Call the main function
if __name__ == "__main__":
    main()
