import streamlit as st
import pandas as pd
from PIL import Image
import pickle

# Set page configuration
st.set_page_config(
    page_title="Crop Prediction System",
    page_icon="🌾",
    layout="wide"
)

# Enhanced CSS styling
st.markdown("""
    <style>
    .main {
        padding: 20px;
        background-color: #f5f5f5;
    }
    .title {
        color: #2e7d32;
        text-align: center;
        padding: 20px;
        font-size: 3em;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background-color: #2e7d32;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        width: 100%;
        font-size: 18px;
    }
    .stButton > button:hover {
        background-color: #1b5e20;
    }
    .stNumberInput > div > div > input {
        border-radius: 5px;
        border: 1px solid #2e7d32;
    }
    .stSubheader {
        color: #1b5e20;
        font-weight: bold;
        font-size: 1.5em;
        margin-bottom: 20px;
    }
    .css-1d391kg {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='title'>Crop Prediction System</h1>", unsafe_allow_html=True)
st.write("Enter soil parameters and environmental conditions to predict the suitable crop")

# Load the trained model and scaler
try:
    model = pickle.load(open('crop_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except Exception as e:
    st.error("Error: Please make sure the model is trained and saved first!")
    st.stop()

# Create two columns for input fields
col1, col2 = st.columns(2)

with col1:
    st.subheader("Soil Parameters")
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50)
    P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=50)
    K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=50)
    ph = st.number_input("pH Level", min_value=3.5, max_value=10.0, value=6.5, step=0.1)

with col2:
    st.subheader("Environmental Conditions")
    temperature = st.number_input("Temperature (°C)", min_value=8.0, max_value=44.0, value=25.0, step=0.1)
    humidity = st.number_input("Humidity (%)", min_value=14.0, max_value=100.0, value=71.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=300.0, value=100.0, step=0.1)

# Prediction button
if st.button("Predict Crop", key="predict"):
    try:
        # Create input data frame
        input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Display result
        st.success(f"Recommended Crop: **{prediction.upper()}**")
        
        # Display confidence scores
        st.subheader("Prediction Confidence Scores")
        probabilities = model.predict_proba(input_scaled)[0]
        crops = model.classes_
        
        # Create a DataFrame for displaying probabilities
        prob_df = pd.DataFrame({
            'Crop': crops,
            'Confidence': probabilities
        }).sort_values(by='Confidence', ascending=False).head(5)
        
        st.dataframe(prob_df.style.format({'Confidence': '{:.2%}'}))
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add information section
with st.expander("About the System"):
    st.write("""
    This Crop Prediction System uses machine learning to recommend suitable crops based on:
    - Soil composition (N, P, K values)
    - Environmental conditions (temperature, humidity, rainfall)
    - Soil pH level
    
    The system uses a Random Forest Classifier trained on agricultural data to make predictions.
    """)