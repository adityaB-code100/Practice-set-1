import streamlit as st
import pandas as pd
import joblib


model = joblib.load('salary_model.joblib')
label_encoders = joblib.load('salary_encoders.joblib')

# Streamlit App UI
st.title("💼 Salary Prediction App")
st.write("Enter the employee details to predict the estimated salary.")

# User Inputs
department = st.selectbox("Department", label_encoders['Department'].classes_)
education = st.selectbox("Education Level", label_encoders['Education Level'].classes_)
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5, step=1)
performance = st.slider("Performance Rating", min_value=1, max_value=5, value=3)

# Prediction Button
if st.button("Predict Salary"):
    # Encode categorical inputs
    encoded_dept = label_encoders['Department'].transform([department])[0]
    encoded_edu = label_encoders['Education Level'].transform([education])[0]

    # Create input dataframe for prediction
    input_df = pd.DataFrame({
        'Department': [encoded_dept],
        'Years Experience': [experience],
        'Education Level': [encoded_edu],
        'Performance Rating': [performance]
    })

    # Predict
    prediction = model.predict(input_df)
    st.success(f"💰 Predicted Salary: **${prediction[0]:.2f}k**")
