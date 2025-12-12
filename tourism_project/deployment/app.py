import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="DeepthiJ28/tourism-package-prediction", filename="best_tourism_package_prediction_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Prediction App")
st.write("""
This application predicts the likelihood of a person taking a tourism package
Please enter the details below get a prediction.
""")

# User input
# Age: 0–120 years
age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1  )

# TypeofContact: Company Invited or Self Inquiry
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])

# CityTier: Tier 1 > Tier 2 > Tier 3
city_tier = st.selectbox("City Tier", ["1", "2", "3"], index=1)

# Occupation
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer", "Freelancer"])

# Gender
gender = st.selectbox("Gender", ["Male", "Female"])

# NumberOfPersonVisiting: 0–20 (including the customer)
number_of_person_visiting = st.number_input("Number of Persons Visiting (including customer)",min_value=0, max_value=20, value=2, step=1)

# PreferredPropertyStar: 1–5
preferred_property_star = st.slider("Preferred Property Star", min_value=1, max_value=5, value=3)

# MaritalStatus
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])

# NumberOfTrips: 0–50 per year
number_of_trips = st.number_input("Average Number of Trips per Year", min_value=0, max_value=50, value=2, step=1)

# Passport: 0/1
passport = st.radio("Passport (0: No, 1: Yes)", options=[0, 1], index=1, horizontal=True )

# OwnCar: 0/1
own_car = st.radio("Own Car (0: No, 1: Yes)", options=[0, 1], index=0, horizontal=True)

# NumberOfChildrenVisiting: 0–10 (children under 5)
number_of_children_visiting = st.number_input("Number of Children Visiting (under age 5)",min_value=0, max_value=10, value=0, step=1)

# Designation
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# MonthlyIncome: currency-neutral numeric
monthly_income = st.number_input("Monthly Income", min_value=0.0, max_value=10_000_000.0, value=50_000.0, step=1_000.0, format="%.2f")

st.subheader("Customer Interaction Data")

# PitchSatisfactionScore: 1–5 or 0–10; using 1–5 here
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)

# ProductPitched
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

# NumberOfFollowups: 0–20
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=20, value=2, step=1)

# DurationOfPitch: minutes, 0–180
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=180, value=15, step=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": typeof_contact,
        "CityTier": city_tier,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_person_visiting,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "ProductPitched": product_pitched,
        "NumberOfFollowups": number_of_followups,
        "DurationOfPitch": duration_of_pitch
}])


if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Tourism package selection" if prediction == 1 else "Not selected"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
