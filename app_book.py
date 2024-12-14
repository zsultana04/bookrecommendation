import streamlit as st
import numpy as np
import joblib

# Load the trained model and label encoders
model = joblib.load('book_category_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Title of the app
st.title("Book Category Predictor")

# Dropdown for Gender
gender = st.selectbox("Select Gender", ['Male', 'Female'])

# Slider for Age
age = st.slider("Select Age", min_value=12, max_value=70, value=25)

# Dropdown for Education Level
education = st.selectbox(
    "Select Education Level",
    ['Below High School', 'High School', 'College', 'Graduate']
)

# Predict button
if st.button("Predict Book Category"):
    try:
        # Encode input features
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        education_encoded = label_encoders['Education'].transform([education])[0]

        # Create input array
        input_data = np.array([[gender_encoded, age, education_encoded]])

        # Make prediction
        prediction = model.predict(input_data)
        predicted_category = label_encoders['BookCategory'].inverse_transform(prediction)[0]

        # Display the result
        st.success(f"Recommended Book Category: **{predicted_category}**")
    except Exception as e:
        st.error(f"Error: {e}")
