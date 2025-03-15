import joblib
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
# using streamlit-extras library features
from streamlit_extras.stylable_container import stylable_container

#load saved model and scaler
model = joblib.load("breast-cancer.pkl")
scaler = joblib.load("breast-cancer-scaler.pkl")

#retrieve feature names from the scaler (if available)
if hasattr(scaler, "feature_names_in_"):
    feature_names = list(scaler.feature_names_in_)
else:
    feature_names = [
        "mean radius", "worst concavity", "mean area", "mean concavity", "mean perimeter",
        "worst perimeter", "worst radius", "mean concave points", "worst concave points", "worst area"
    ]
#load and resize the image
image = Image.open("cancer.jpg")
resized_image = image.resize((image.width, 200))

st.image(resized_image, use_container_width=True)

st.title("Breast Cancer Prediction")
st.markdown("Enter the feature values below to predict whether a tumor is benign or malignant.")

#taking user input 
def get_user_input():
    user_data = {}
    with st.expander("Enter Feature Values"):
        for feature in feature_names:
            user_data[feature] = st.text_input(f"Enter value for {feature}", "0.0")
    df = pd.DataFrame([user_data])
    return df

input_df = get_user_input()
#displaying entered data into table format
st.subheader("Your Data :")
st.data_editor(input_df, num_rows="fixed", use_container_width=True)

#predicting 
if st.button("Predict", use_container_width=True):
    try:
        #converting input data to numpy array and ensure correct data type
        input_array = input_df.astype(float).to_numpy().reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)
        
        #display results with a styled container
        with stylable_container("prediction-box", css_styles="background-color: #f0f0f5; padding: 50px; border-radius: 10px; text-align: center;"):
            st.write("## **Malignant**" if prediction[0] == 0 else "##  **Benign**")
            
            st.subheader("Prediction Probability :")
            st.write(f" **Malignant:** {prediction_proba[0][0] * 100:.2f}%")
            st.write(f" **Benign:** {prediction_proba[0][1] * 100:.2f}%")
    except Exception as e:
        st.error(f"Error: {e}")