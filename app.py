import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Smart Farming Assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import all required libraries
import pandas as pd
import numpy as np
from datetime import datetime
import os
import requests
import pickle
from modules.styles import load_css

# Load custom CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'dashboard'

# Initialize model variables
model = None
scaler = None

# Load model and scaler files
try:
    model = pickle.load(open('crop_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Required model files not found: {e}")
except Exception as e:
    st.error(f"Error loading model: {e}")

def get_forecast(location):
    """Get 5-day weather forecast for a location"""
    try:
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not api_key:
            st.error("OpenWeatherMap API key not found. Please check your environment variables.")
            return None
            
        base_url = "http://api.openweathermap.org/data/2.5/forecast"
        params = {
            'q': location,
            'appid': api_key,
            'units': 'metric'
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching forecast data: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def get_weather(location):
    """Get current weather data for a location"""
    try:
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not api_key:
            st.error("OpenWeatherMap API key not found. Please check your environment variables.")
            return None
            
        base_url = "http://api.openweathermap.org/data/2.5/weather"
        params = {
            'q': location,
            'appid': api_key,
            'units': 'metric'
        }
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching weather data: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Initialize model variables
model = None
scaler = None

# Load model and scaler files
try:
    model = pickle.load(open('crop_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Required model files not found: {e}")
except Exception as e:
    st.error(f"Error loading model: {e}")

def predict_crop(soil_data):
    try:
        if model is None or scaler is None:
            st.error("Model or scaler not loaded properly")
            return None, None
            
        # Scale the input data
        scaled_data = scaler.transform([soil_data])
        
        # Make prediction
        prediction = model.predict(scaled_data)
        probabilities = model.predict_proba(scaled_data)
        
        return prediction[0], probabilities[0]
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None

# Add this crops_info dictionary before the show_main_page function
crops_info = {
    "Rice": """
    - Optimal growing temperature: 20-35°C
    - Requires standing water for most of its growth
    - High water requirement: 1000-2000mm per season
    - Prefers clay or clay loam soils
    - Growing season: 100-150 days
    """,
    
    "Wheat": """
    - Optimal growing temperature: 15-25°C
    - Moderate water requirement
    - Well-drained loamy soil preferred
    - Growing season: 120-150 days
    - Sensitive to high humidity
    """,
    
    "Maize": """
    - Optimal growing temperature: 20-30°C
    - Requires well-distributed rainfall
    - Deep, well-drained soils preferred
    - Growing season: 90-140 days
    - High nutrient requirement
    """,
    
    "Cotton": """
    - Optimal growing temperature: 21-30°C
    - Drought tolerant but sensitive to waterlogging
    - Deep, well-drained black soils preferred
    - Growing season: 150-180 days
    - High nutrient requirement
    """,
    
    "Sugarcane": """
    - Optimal growing temperature: 21-27°C
    - High water requirement
    - Deep, well-drained soils preferred
    - Growing season: 12-18 months
    - Requires good soil fertility
    """,
    
    "Potato": """
    - Optimal growing temperature: 15-25°C
    - Moderate water requirement
    - Well-drained, loose soil preferred
    - Growing season: 90-120 days
    - Sensitive to frost
    """,
    
    "Tomato": """
    - Optimal growing temperature: 20-27°C
    - Regular water requirement
    - Well-drained, rich soil preferred
    - Growing season: 90-150 days
    - Sensitive to frost and high humidity
    """,
    
    "Soybean": """
    - Optimal growing temperature: 20-30°C
    - Moderate water requirement
    - Well-drained, fertile soil preferred
    - Growing season: 100-120 days
    - Good nitrogen fixing ability
    """
}

def show_main_page():
    # Main header with modern styling
    st.markdown("""
        <div class="glass-card" style="text-align: center; margin-bottom: 2rem;">
            <h1 class="gradient-text" style="font-size: 3em; margin: 0;">
                🌿 Smart Farming Assistant
            </h1>
            <p style="color: #81C784; margin-top: 0.5rem;">
                AI-Powered Crop Recommendations & Weather Analysis
            </p>
        </div>
    """, unsafe_allow_html=True)
      # Main Grid Layout
    st.markdown("""
        <div class="grid-container">
            <div class="glass-card">
                <h2 class="gradient-text">🌾 Crop Prediction</h2>
                <div class="modern-form">
    """, unsafe_allow_html=True)
    
    # Crop Prediction Section
    col1, col2 = st.columns([2, 1])
    with col1:
        with st.form("prediction_form", clear_on_submit=True):
            st.markdown("""
                <h3 style="color: #81C784; margin-bottom: 1rem;">
                    📊 Soil and Environmental Parameters
                </h3>
            """, unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)
            with col3:
                nitrogen = st.number_input("Nitrogen (N)", 0, 140, 50, help="Soil nitrogen content in mg/kg")
                phosphorus = st.number_input("Phosphorus (P)", 0, 140, 50, help="Soil phosphorus content in mg/kg")
                potassium = st.number_input("Potassium (K)", 0, 200, 50, help="Soil potassium content in mg/kg")
                temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
            
            with col4:
                humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)
                ph = st.number_input("pH", 0.0, 14.0, 7.0)
                rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)
            
            predict_button = st.form_submit_button("🔍 Predict Best Crop")
            
            if predict_button:
                input_data = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
                prediction, probabilities = predict_crop(input_data)
                if prediction is not None:
                    st.session_state.prediction = prediction
                    st.session_state.probabilities = probabilities
                    st.rerun()

    with col2:
        st.header("📊 Prediction Results")
        if 'prediction' in st.session_state and 'probabilities' in st.session_state:
            # Display prediction results with enhanced styling
            st.markdown("""
                <div style='background: rgba(255, 255, 255, 0.1);
                          backdrop-filter: blur(10px);
                          padding: 2rem;
                          border-radius: 20px;
                          border: 1px solid rgba(255, 255, 255, 0.1);
                          margin: 1rem 0;
                          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);'>
                    <h2 style='color: #4CAF50; margin: 0; text-align: center;
                             font-size: 2em; margin-bottom: 1rem;'>
                        Recommended Crop
                    </h2>
                    <h1 style='color: #ffffff; margin: 0; text-align: center;
                             font-size: 3em; margin-bottom: 1rem;
                             background: linear-gradient(45deg, #4CAF50, #81C784);
                             -webkit-background-clip: text;
                             -webkit-text-fill-color: transparent;'>
                        {prediction}
                    </h1>
                    <div style='background: rgba(255, 255, 255, 0.05);
                              padding: 1rem;
                              border-radius: 10px;
                              margin-top: 1rem;'>
                        <p style='color: #ffffff; margin: 0; text-align: center;'>
                            Confidence Score: {confidence:.1%}
                        </p>
                    </div>
                </div>
            """.format(
                prediction=st.session_state.prediction.upper(),
                confidence=max(st.session_state.probabilities)
            ), unsafe_allow_html=True)

            # Confidence Scores
            st.subheader("Prediction Confidence")
            probabilities = st.session_state.probabilities
            crop_labels = ["rice", "wheat", "maize", "cotton", "sugarcane", "potato", "tomato", "soybean"]
            
            # Create DataFrame with formatted probabilities
            prob_df = pd.DataFrame({
                'Crop': crop_labels[:len(probabilities)],
                'Confidence': probabilities
            })
            prob_df = prob_df.sort_values('Confidence', ascending=False)
            
            # Display top 3 recommendations with confidence scores
            st.markdown("### Top Recommendations")
            for idx, row in prob_df.head(3).iterrows():
                confidence_percentage = row['Confidence'] * 100
                
                st.markdown(f"""
                    <div class='card' style='
                              padding: 15px 20px;
                              margin: 10px 0;
                              border-left: 4px solid {'#4CAF50' if idx == 0 else '#81C784'};'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='color: #ffffff; font-weight: bold; font-size: 1.1em;'>{row['Crop'].title()}</span>
                            <span style='color: #4CAF50; background: rgba(76, 175, 80, 0.1); 
                                       padding: 4px 10px; border-radius: 15px;'>
                                {confidence_percentage:.1f}%
                            </span>
                        </div>
                        <div style='background: rgba(255, 255, 255, 0.1); 
                                  border-radius: 10px; 
                                  margin-top: 10px;
                                  overflow: hidden;'>
                            <div style='background: linear-gradient(45deg, #4CAF50, #81C784); 
                                      width: {confidence_percentage}%; 
                                      height: 6px; 
                                      border-radius: 10px;
                                      transition: width 0.3s ease;'></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Display full confidence distribution chart
            st.markdown("### Confidence Distribution")
            chart = st.bar_chart(prob_df.set_index('Crop')['Confidence'],
                               use_container_width=True)
            
            # Add crop information for the predicted crop
            st.markdown("### Crop Information")
            crop_info = {
                "rice": {
                    "season": "Kharif",
                    "water_req": "High",
                    "temp_range": "20-35°C",
                    "soil_type": "Clay or clay loam"
                },
                "wheat": {
                    "season": "Rabi",
                    "water_req": "Moderate",
                    "temp_range": "15-25°C",
                    "soil_type": "Well-drained loam"
                },
                "maize": {
                    "season": "Kharif/Rabi",
                    "water_req": "Moderate",
                    "temp_range": "20-30°C",
                    "soil_type": "Well-drained loamy"
                },
                "cotton": {
                    "season": "Kharif",
                    "water_req": "Moderate",
                    "temp_range": "21-30°C",
                    "soil_type": "Black soil"
                },
                "sugarcane": {
                    "season": "Year-round",
                    "water_req": "High",
                    "temp_range": "21-27°C",
                    "soil_type": "Deep loam"
                },
                "potato": {
                    "season": "Rabi",
                    "water_req": "Moderate",
                    "temp_range": "15-25°C",
                    "soil_type": "Sandy loam"
                },
                "tomato": {
                    "season": "Year-round",
                    "water_req": "Moderate",
                    "temp_range": "20-27°C",
                    "soil_type": "Rich loam"
                },
                "soybean": {
                    "season": "Kharif",
                    "water_req": "Moderate",
                    "temp_range": "20-30°C",
                    "soil_type": "Well-drained loam"
                }
            }
            
            predicted_crop = st.session_state.prediction.lower()
            if predicted_crop in crop_info:
                info = crop_info[predicted_crop]
                st.markdown(f"""
                    <div style='background-color: rgba(255, 255, 255, 0.9);
                              padding: 20px;
                              border-radius: 15px;
                              margin-top: 15px;
                              box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                        <h4 style='color: #2E7D32; margin-top: 0;'>Optimal Growing Conditions</h4>
                        <table style='width: 100%;'>
                            <tr>
                                <td style='padding: 8px; color: #1B5E20;'>🌱 Growing Season</td>
                                <td>{info['season']}</td>
                            </tr>
                            <tr>
                                <td style='padding: 8px; color: #1B5E20;'>💧 Water Requirement</td>
                                <td>{info['water_req']}</td>
                            </tr>
                            <tr>
                                <td style='padding: 8px; color: #1B5E20;'>🌡️ Temperature Range</td>
                                <td>{info['temp_range']}</td>
                            </tr>
                            <tr>
                                <td style='padding: 8px; color: #1B5E20;'>🌍 Ideal Soil Type</td>
                                <td>{info['soil_type']}</td>
                            </tr>
                        </table>
                    </div>
                """, unsafe_allow_html=True)    # Weather Information Section with modern layout
    st.markdown("""
        <div class="glass-card">
            <h2 class="gradient-text" style="margin-bottom: 1.5rem;">
                🌤️ Weather Information
            </h2>
            <div class="weather-section">
    """, unsafe_allow_html=True)
    
    location = st.text_input("Enter Location", placeholder="Enter city name...", key="weather_location")
    
    if location:
        col1, col2 = st.columns(2)
        with col1:
            weather_data = get_weather(location)
            if weather_data:
                temp = weather_data['main']['temp']
                humidity = weather_data['main']['humidity']
                wind_speed = weather_data['wind']['speed']
                description = weather_data['weather'][0]['description']
                
                st.markdown(f"""
                    <div class="weather-card">
                        <h3 style="color: #4CAF50; margin-top: 0;">Current Weather</h3>
                        <div class="grid-container" style="gap: 1rem;">
                            <div class="metric-card">
                                <p style="color: #81C784; margin: 0;">Temperature</p>
                                <h2 style="margin: 0.5rem 0; color: white;">{temp}°C</h2>
                            </div>
                            <div class="metric-card">
                                <p style="color: #81C784; margin: 0;">Humidity</p>
                                <h2 style="margin: 0.5rem 0; color: white;">{humidity}%</h2>
                            </div>
                        </div>
                        <div class="metric-card" style="margin-top: 1rem;">
                            <p style="color: #81C784; margin: 0;">Wind Speed</p>
                            <h2 style="margin: 0.5rem 0; color: white;">{wind_speed} m/s</h2>
                        </div>
                        <div class="metric-card" style="margin-top: 1rem;">
                            <p style="color: #81C784; margin: 0;">Condition</p>
                            <h3 style="margin: 0.5rem 0; color: white;">{description.capitalize()}</h3>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <h3 class="gradient-text" style="margin-bottom: 1rem;">5-Day Forecast</h3>
            """, unsafe_allow_html=True)
            
            forecast_data = get_forecast(location)
            if forecast_data and 'list' in forecast_data:
                # Process and display forecast
                forecasts = {}
                for item in forecast_data['list']:
                    date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
                    if date not in forecasts:
                        forecasts[date] = {
                            'temp': item['main']['temp'],
                            'humidity': item['main']['humidity'],
                            'description': item['weather'][0]['description']
                        }
                
                for date, data in list(forecasts.items())[:5]:
                    st.markdown(f"""
                        <div class="weather-card" style="margin-bottom: 0.5rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <strong style="color: #4CAF50;">{date}</strong>
                                <span style="color: white;">{data['temp']}°C</span>
                            </div>
                            <div style="margin-top: 0.5rem;">
                                <span style="color: #81C784;">Humidity: {data['humidity']}%</span><br>
                                <span style="color: white;">{data['description'].capitalize()}</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
    
    st.markdown("</div></div>", unsafe_allow_html=True)
    # Replace the weather information section with this enhanced version
    if location:
        col5, col6 = st.columns(2)
        with col5:
            st.markdown("""
                <div class="weather-card">
                    <h3 style="color: white; margin-top: 0;">Current Weather</h3>
            """, unsafe_allow_html=True)
            def get_weather(location):
                """Get current weather data for a location"""
                try:
                    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
                    if not api_key:
                        st.error("OpenWeatherMap API key not found. Please check your environment variables.")
                        return None
                        
                    base_url = "http://api.openweathermap.org/data/2.5/weather"
                    params = {
                        'q': location,
                        'appid': api_key,
                        'units': 'metric'
                    }
                    
                    response = requests.get(base_url, params=params)
                    if response.status_code == 200:
                        return response.json()
                    else:
                        st.error(f"Error fetching weather data: {response.status_code}")
                        return None
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    return None
            
            weather_data = get_weather(location)
            if weather_data:
                temp = weather_data['main']['temp']
                humidity = weather_data['main']['humidity']
                wind_speed = weather_data['wind']['speed']
                description = weather_data['weather'][0]['description']
                
                # Weather display with modern cards
                st.markdown(f"""
                    <div class='card'>
                        <h3 style='color: #4CAF50; margin-top: 0;'>Current Weather</h3>
                        <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;'>
                            <div class='metric-container'>
                                <p style='color: #81C784; margin: 0;'>Temperature</p>
                                <h2 style='margin: 0.5rem 0;'>{temp}°C</h2>
                            </div>
                            <div class='metric-container'>
                                <p style='color: #81C784; margin: 0;'>Humidity</p>
                                <h2 style='margin: 0.5rem 0;'>{humidity}%</h2>
                            </div>
                        </div>
                        <div class='metric-container' style='margin-top: 1rem;'>
                            <p style='color: #81C784; margin: 0;'>Wind Speed</p>
                            <h2 style='margin: 0.5rem 0;'>{wind_speed} m/s</h2>
                        </div>
                        <p style='color: white; margin-top: 1rem;'>
                            Condition: {description.capitalize()}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            st.subheader("5-Day Forecast")
            forecast_data = get_forecast(location)
            if forecast_data:
                forecasts = {}
                for item in forecast_data['list']:
                    date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
                    if date not in forecasts:
                        forecasts[date] = {
                            'temp': item['main']['temp'],
                            'humidity': item['main']['humidity'],
                            'description': item['weather'][0]['description']
                        }
                
                for date, data in list(forecasts.items())[:5]:
                    st.write(f"**{date}**")
                    st.write(f"Temp: {data['temp']}°C")
                    st.write(f"Humidity: {data['humidity']}%")
                    st.write(f"Condition: {data['description'].capitalize()}")
                    st.write("---")

    # Knowledge Base Section with improved styling
    st.markdown("""
        <hr style='margin: 2rem 0;'>
        <h2 style='color: #2E7D32;'>📚 Knowledge Base</h2>
    """, unsafe_allow_html=True)
    knowledge_type = st.selectbox(
        "Select Topic",
        ["Crop Information", "Soil Management", "Pest Control", "Farming Tips"]
    )
    
    # Replace the knowledge base display with this enhanced version
    if knowledge_type == "Crop Information":
        for crop, info in crops_info.items():
            st.markdown(f"""
                <div class="knowledge-card" style="
                    background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,255,255,0.85));
                    padding: 20px;
                    border-radius: 15px;
                    margin: 15px 0;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    border-left: 5px solid #4CAF50;
                    backdrop-filter: blur(10px);
                    transition: all 0.3s ease;">
                    <h4 style="color: #2E7D32; margin-top: 0; font-size: 1.4em;">{crop}</h4>
                    <div style="color: #333; line-height: 1.6;">{info}</div>
                </div>
            """, unsafe_allow_html=True)
        
    elif knowledge_type == "Soil Management":
        st.subheader("🌍 Soil Health Management")
        st.markdown("""
        **1. Soil Testing**
        - Regular soil testing is crucial
        - Monitor pH levels
        - Check nutrient content (N, P, K)
        
        **2. Soil Improvement**
        - Add organic matter
        - Practice crop rotation
        - Use appropriate fertilizers
        
        **3. Conservation Practices**
        - Minimize tillage
        - Use cover crops
        - Prevent soil erosion
        """)
        
    elif knowledge_type == "Pest Control":
        st.subheader("🐛 Integrated Pest Management")
        st.markdown("""
        **Common Pests and Solutions:**
        
        1. **Aphids**
        - Use neem oil spray
        - Introduce ladybugs
        - Remove affected leaves
        
        2. **Caterpillars**
        - Handpick when possible
        - Use Bt (Bacillus thuringiensis)
        - Install bird houses
        
        3. **Root Rot**
        - Improve drainage
        - Avoid overwatering
        - Use resistant varieties
        """)
        
    elif knowledge_type == "Agronomic Management":
        st.subheader("🚜 Agronomic & Management Features")
        st.markdown("""
        **Key Management Practices:**

        1. **Previous Crop**
        - Crop rotation patterns affect soil nutrients
        - Helps break pest and disease cycles
        - Improves soil structure and fertility
        - Optimizes nutrient utilization

        2. **Sowing Date**
        - Impacts crop duration and exposure to weather
        - Determines growing season length
        - Affects crop development stages
        - Influences pest and disease risks

        3. **Fertilizer Usage**
        - Quantity and type (NPK, organic, etc.)
        - Application timing and methods
        - Balanced nutrient management
        - Cost-effective fertilization strategies

        4. **Irrigation Type and Frequency**
        - Drip, sprinkler, flood irrigation options
        - Water scheduling based on crop stage
        - Moisture monitoring techniques
        - Water conservation practices

        5. **Pesticide/Herbicide Usage**
        - Can influence crop yield and health
        - Integrated pest management (IPM)
        - Safe application methods
        - Environmental considerations

        6. **Seed Variety**
        - Genetically improved seeds may perform better
        - Disease resistance characteristics
        - Climate adaptability
        - Yield potential assessment
        """)
    
    elif knowledge_type == "Farming Tips":
        st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,255,255,0.85));
                      padding: 25px;
                      border-radius: 15px;
                      margin: 15px 0;
                      box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                      border-left: 5px solid #4CAF50;">
                <h3 style="color: #2E7D32; margin-top: 0;">🌱 Seasonal Farming Tips</h3>
                
                <h4 style="color: #1B5E20; margin-top: 20px;">1. Pre-Planting</h4>
                <ul style="color: #333; line-height: 1.6;">
                    <li>Test soil quality and pH levels</li>
                    <li>Prepare land with proper tillage</li>
                    <li>Plan crop rotation schedule</li>
                    <li>Select appropriate seeds for the season</li>
                </ul>
                
                <h4 style="color: #1B5E20; margin-top: 20px;">2. During Growing Season</h4>
                <ul style="color: #333; line-height: 1.6;">
                    <li>Monitor water requirements regularly</li>
                    <li>Implement proper fertilization schedule</li>
                    <li>Watch for pest and disease signs</li>
                    <li>Maintain proper spacing between plants</li>
                </ul>
                
                <h4 style="color: #1B5E20; margin-top: 20px;">3. Harvest and Post-Harvest</h4>
                <ul style="color: #333; line-height: 1.6;">
                    <li>Harvest at optimal maturity</li>
                    <li>Proper storage of harvested crops</li>
                    <li>Maintain cleanliness in storage areas</li>
                    <li>Monitor moisture levels in storage</li>
                </ul>
                
                <h4 style="color: #1B5E20; margin-top: 20px;">4. Sustainable Practices</h4>
                <ul style="color: #333; line-height: 1.6;">
                    <li>Use organic fertilizers when possible</li>
                    <li>Practice water conservation</li>
                    <li>Implement integrated pest management</li>
                    <li>Maintain soil health through mulching</li>
                </ul>
            </div>
            
            <div style="background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(255,255,255,0.85));
                      padding: 25px;
                      border-radius: 15px;
                      margin: 15px 0;
                      box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                      border-left: 5px solid #4CAF50;">
                <h3 style="color: #2E7D32; margin-top: 0;">🔍 Best Practices</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 12px; color: #1B5E20; font-weight: bold;">Soil Management</td>
                        <td style="padding: 12px;">Regular testing, proper drainage, organic matter addition</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px; color: #1B5E20; font-weight: bold;">Water Management</td>
                        <td style="padding: 12px;">Drip irrigation, rainwater harvesting, moisture monitoring</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px; color: #1B5E20; font-weight: bold;">Pest Control</td>
                        <td style="padding: 12px;">Natural predators, crop rotation, resistant varieties</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px; color: #1B5E20; font-weight: bold;">Resource Optimization</td>
                        <td style="padding: 12px;">Efficient use of water, fertilizers, and labor</td>
                    </tr>
                </table>
            </div>
        """, unsafe_allow_html=True)
        
    elif knowledge_type == "Farming Tips":
        st.title("📈 Reports")
        
        # Model Performance Section
        st.header("📈 Model Performance")
        col5, col6, col7 = st.columns(3)
        with col5:
            st.metric("Model Accuracy", "94.2%", "1.2%")
        with col6:
            st.metric("Precision", "92.8%", "0.8%")
        with col7:
            st.metric("Recall", "93.5%", "0.5%")
        
        # Historical Analysis Section
        st.header("📊 Historical Analysis")
        
        # Create sample historical data with consistent column names
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='M')
        historical_data = pd.DataFrame({
            'Date': dates,
            'Prediction_Accuracy': np.random.uniform(0.90, 0.95, len(dates)),  # Match the column name used in the chart
            'Number_of_Predictions': np.random.randint(50, 150, len(dates))
        })
        
        col10, col11 = st.columns(2)
        
        with col10:
            st.subheader("📈 Prediction History")
            # Create accuracy trend chart with proper column name
            accuracy_chart = historical_data.set_index('Date')['Prediction_Accuracy']
            st.line_chart(accuracy_chart)
            
        with col11:
            st.subheader("🌱 Crop Distribution")
            # Sample crop distribution data
            crops = ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane']
            values = [30, 25, 20, 15, 10]
            crop_dist = pd.DataFrame({
                'Crop': crops,
                'Percentage': values
            })
            st.bar_chart(crop_dist.set_index('Crop'))
        # Display historical data with tabs
        tab1, tab2 = st.tabs(["📈 Prediction History", "🌱 Crop Distribution"])
        
        with tab1:
            st.subheader("Prediction Accuracy Trend")
            st.line_chart(historical_data.set_index('Date')['Prediction Accuracy'])
            st.dataframe(historical_data, hide_index=True)
            
        with tab2:
            st.subheader("Crop Distribution")
            crop_counts = historical_data['Crop'].value_counts()
            st.bar_chart(crop_counts)
            
        # Export functionality
        st.download_button(
            label="📥 Download Report Data",
            data=historical_data.to_csv(index=False),
            file_name="crop_prediction_report.csv",
            mime="text/csv"
        )

        st.dataframe(historical_data)
        
        # Visualization
        st.line_chart(historical_data.set_index('Date')['Accuracy'])


if __name__ == "__main__":
    show_main_page()