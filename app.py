import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import json
import os
import requests
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from test_model import predict_crop  # Add this import

def get_model_info():
    return {
        'feature_importance': {
            'Rainfall': {'value': 18.36, 'ci_lower': 17.2, 'ci_upper': 19.5},
            'Humidity': {'value': 17.89, 'ci_lower': 16.8, 'ci_upper': 19.0},
            'Nitrogen': {'value': 16.45, 'ci_lower': 15.3, 'ci_upper': 17.6},
            'pH': {'value': 15.51, 'ci_lower': 14.4, 'ci_upper': 16.6},
            'Potassium': {'value': 14.11, 'ci_lower': 13.0, 'ci_upper': 15.2},
            'Temperature': {'value': 9.73, 'ci_lower': 8.6, 'ci_upper': 10.9},
            'Phosphorus': {'value': 7.95, 'ci_lower': 6.8, 'ci_upper': 9.1}
        },
        'performance': {
            'accuracy': 0.90,
            'cv_score': 0.885,
            'precision': 0.88,
            'recall': 0.87,
            'f1_score': 0.89,
            'model_type': 'Random Forest Classifier',
            'last_updated': '2024-01-20',
            'training_samples': 10000,
            'validation_method': '5-fold cross validation'
        },
        'recommendations': {
            'Rainfall': {'low': 50, 'optimal': 200, 'high': 300},
            'Humidity': {'low': 30, 'optimal': 70, 'high': 90},
            'Temperature': {'low': 10, 'optimal': 25, 'high': 35},
            'pH': {'low': 5.5, 'optimal': 6.5, 'high': 7.5},
            'Nitrogen': {'low': 40, 'optimal': 90, 'high': 120},
            'Phosphorus': {'low': 20, 'optimal': 42, 'high': 80},
            'Potassium': {'low': 20, 'optimal': 43, 'high': 100}
        },
        'model_version': '2.0',
        'update_frequency': 'Monthly'
    }

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

st.set_page_config(page_title="Advanced Crop Recommendation System", layout="wide")

# Update the Custom CSS with ultra-modern styling and dark theme support
st.markdown("""
    <style>
    /* Dark theme variables */
    :root {
        --primary-color: #4CAF50;
        --primary-light: #81C784;
        --primary-dark: #2E7D32;
        --text-color: #1B5E20;
        --bg-gradient: linear-gradient(135deg, #f5f9f5 0%, #ffffff 100%);
        --card-bg: rgba(255, 255, 255, 0.95);
    }

    /* Dark theme styles */
    @media (prefers-color-scheme: dark) {
        :root {
            --primary-color: #81C784;
            --primary-light: #A5D6A7;
            --primary-dark: #4CAF50;
            --text-color: #E8F5E9;
            --bg-gradient: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            --card-bg: rgba(45, 45, 45, 0.95);
        }
    }

    .main {
        padding: 45px;
        background: var(--bg-gradient);
        min-height: 100vh;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary-color) 100%);
        color: white;
        border-radius: 35px;
        border: none;
        padding: 24px 48px;
        transition: all 0.8s cubic-bezier(0.22, 1, 0.36, 1);
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-size: 1.3em;
        box-shadow: 0 15px 35px rgba(76, 175, 80, 0.3);
        backdrop-filter: blur(30px);
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-dark) 100%);
        box-shadow: 0 20px 45px rgba(76, 175, 80, 0.4);
        transform: translateY(-10px) scale(1.03);
    }

    .css-1d391kg {
        padding: 4.5rem;
        border-radius: 45px;
        background: var(--card-bg);
        box-shadow: 0 30px 70px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(50px);
        margin: 4rem 0;
        border: 1px solid rgba(76, 175, 80, 0.18);
        transition: all 0.8s ease;
    }

    /* Add new modern card effect */
    .card-effect {
        position: relative;
        overflow: hidden;
    }

    .card-effect::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(129, 199, 132, 0.2), rgba(76, 175, 80, 0.1));
        transform: translateY(100%);
        transition: transform 0.6s cubic-bezier(0.22, 1, 0.36, 1);
        z-index: -1;
        border-radius: inherit;
    }

    .card-effect:hover::before {
        transform: translateY(0);
    }

    /* Enhanced metric styling */
    .stMetric {
        background: var(--card-bg);
        padding: 50px;
        border-radius: 40px;
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.12);
        transition: all 0.9s cubic-bezier(0.22, 1, 0.36, 1);
        border: 1px solid rgba(76, 175, 80, 0.18);
        transform-origin: center;
    }

    .stMetric:hover {
        transform: translateY(-12px) scale(1.05);
        box-shadow: 0 30px 60px rgba(0, 0, 0, 0.18);
    }

    /* Improved text styling */
    h1 {
        background: linear-gradient(45deg, var(--primary-dark), var(--primary-color), var(--primary-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 55px 0;
        font-weight: 900;
        letter-spacing: 3px;
        font-size: 4.2em;
        text-shadow: 5px 5px 10px rgba(0,0,0,0.12);
    }
    </style>
    """, unsafe_allow_html=True)

# Enhanced decorative header with modern glassmorphism
st.markdown("""
    <div style='text-align: center; padding: 90px; background: linear-gradient(165deg, #e8f5e9 0%, #c8e6c9 50%, #e8f5e9 100%); border-radius: 50px; margin: 3.5rem 0 5.5rem 0; box-shadow: 0 35px 90px rgba(76, 175, 80, 0.2); border: 1px solid rgba(76, 175, 80, 0.15); backdrop-filter: blur(40px);' class='card-effect'>
        <h1 style='margin-bottom: 4.5rem; text-shadow: 6px 6px 12px rgba(0,0,0,0.15); font-size: 4.5em;'>
            🌾 Advanced Crop Recommendation System
        </h1>
        <p style='font-size: 2.2em; color: var(--text-color); line-height: 2.3; max-width: 1400px; margin: 0 auto; font-weight: 500; text-shadow: 2px 2px 4px rgba(255,255,255,0.95);'>
            An AI-powered system for optimal crop recommendations based on soil and environmental parameters
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main title with enhanced styling
st.title("🌾 Advanced Crop Recommendation System")
st.write("An AI-powered system for optimal crop recommendations based on soil and environmental parameters")

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Prediction", "Analysis", "History", 
    "AI Insights", "Help", "Crop Calendar"
])

# Initialize weather data in session state
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None

# Default environmental values
temperature = 20.87  # Default value
humidity = 82.0     # Default value

with tab1:
    st.subheader("📍 Location-based Weather")
    col_loc1, col_loc2, col_loc3 = st.columns(3)
    
    with col_loc1:
        location = st.text_input("Enter location for weather data", "")
    with col_loc2:
        units = st.selectbox("Temperature Units", ["Celsius", "Fahrenheit"])
    with col_loc3:
        store_weather = st.checkbox("Use weather data for prediction", value=True)
        
    if location and st.button("Fetch Weather"):
        try:
            # Initialize geolocator with a custom user agent
            geolocator = Nominatim(user_agent="crop_recommendation_app_v1")
            location_data = geolocator.geocode(location, timeout=15)
            
            if location_data:
                # Update with your OpenWeatherMap API key
                api_key = "62dfad0e5e6803ade47eec8332c7d006"
                
                unit_param = "imperial" if units == "Fahrenheit" else "metric"
                
                try:
                    weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={location_data.latitude}&lon={location_data.longitude}&appid={api_key}&units={unit_param}"
                    response = requests.get(weather_url, timeout=10)
                    response.raise_for_status()  # This will raise an exception for HTTP errors
                    weather_data = response.json()
                    
                    if store_weather:
                        st.session_state.weather_data = weather_data
                    
                    # Display location and weather information
                    st.success(f"📍 Location Found: {location_data.address}")
                    
                    # Display current conditions in columns
                    # Replace the weather metrics display section with:
                    if weather_data:
                        # Create a modern weather card
                        st.markdown("""
                            <div style='background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
                                 padding: 30px; border-radius: 25px; box-shadow: 0 8px 32px rgba(76, 175, 80, 0.1);
                                 backdrop-filter: blur(8px); border: 1px solid rgba(76, 175, 80, 0.2);'>
                            <h3 style='color: #2E7D32; margin-bottom: 20px;'>Current Weather Conditions</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Display weather metrics in columns
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Temperature", 
                                    f"{weather_data['main']['temp']}°{'F' if units == 'Fahrenheit' else 'C'}",
                                    f"{weather_data['main']['temp_max'] - weather_data['main']['temp']}°")
                        with col2:
                            st.metric("Humidity", 
                                    f"{weather_data['main']['humidity']}%",
                                    "Optimal" if 60 <= weather_data['main']['humidity'] <= 80 else "Sub-optimal")
                        with col3:
                            st.metric("Wind", 
                                    f"{weather_data['wind']['speed']} m/s",
                                    f"Direction: {weather_data['wind'].get('deg', 'N/A')}°")
                        with col4:
                            st.metric("Conditions",
                                    weather_data['weather'][0]['description'].title(),
                                    weather_data['weather'][0]['main'])
                        
                        # Remove the duplicate metrics display
                        # Delete these lines:
                        # with col2:
                        #     st.metric("Humidity", f"{weather_data['main']['humidity']}%")
                        # with col3:
                        #     st.metric("Wind", f"{weather_data['wind']['speed']} m/s")
                        # with col4:
                        #     st.metric("Conditions", weather_data['weather'][0]['description'].title())
                    st.markdown("</div>", unsafe_allow_html=True)
                    with col2:
                        st.metric("Humidity", f"{weather_data['main']['humidity']}%")
                    with col3:
                        st.metric("Wind", f"{weather_data['wind']['speed']} m/s")
                    with col4:
                        st.metric("Conditions", weather_data['weather'][0]['description'].title())
                    
                    # Update environmental parameters if checkbox is selected
                    if store_weather:
                        temperature = weather_data['main']['temp']
                        if units == "Fahrenheit":
                            temperature = (temperature - 32) * 5/9  # Convert to Celsius
                        humidity = weather_data['main']['humidity']
                    
                except requests.exceptions.RequestException as e:
                    if "401" in str(e):
                        st.error("Invalid API key. Please check your OpenWeatherMap API key.")
                        st.info("Get your API key from: https://openweathermap.org/api")
                    else:
                        st.error("Failed to fetch weather data. Please check your internet connection.")
                        st.info("If the error persists, try again in a few minutes.")
            
            else:
                st.error("Location not found! Please try:")
                st.info("""
                - Check the spelling
                - Add more details (city, state, country)
                - Try a nearby major city
                """)
                
        except Exception as e:
            st.error("Error accessing location service. Please try again later.")
            st.info("Make sure you have a stable internet connection")

    st.subheader("📊 Soil Parameters")
    
    # Add soil type selection with enhanced styling
    st.markdown("""
        <style>
        .soil-info {
            padding: 15px;
            border-radius: 10px;
            background: linear-gradient(135deg, rgba(129, 199, 132, 0.1), rgba(76, 175, 80, 0.05));
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    soil_type = st.selectbox("Select Soil Type", ["Alluvial", "Black", "Red", "Laterite", "Mountain", "Desert"])
    soil_based_crops = {
        "Alluvial": ["Rice", "Wheat", "Sugarcane"],
        "Black": ["Cotton", "Soybean", "Groundnut"],
        "Red": ["Millet", "Pulses", "Oilseeds"],
        "Laterite": ["Tea", "Coffee", "Cashew"],
        "Mountain": ["Tea", "Apples", "Spices"],
        "Desert": ["Barley", "Pearl Millet", "Guar"]
    }
    
    if soil_type:
        st.markdown(f"""
            <div class="soil-info">
                <h4>🌱 Recommended Crops for {soil_type} Soil:</h4>
                <p style='font-size: 1.1em; margin-top: 10px;'>{', '.join(soil_based_crops[soil_type])}</p>
            </div>
        """, unsafe_allow_html=True)

    # Existing soil parameter sliders
    N = st.slider("Nitrogen (N) mg/kg", 0, 140, 90, help="Amount of Nitrogen in soil")
    
    P = st.slider("Phosphorus (P) mg/kg", 5, 145, 42, help="Amount of Phosphorus in soil")
    K = st.slider("Potassium (K) mg/kg", 5, 205, 43, help="Amount of Potassium in soil")
    ph = st.slider("Soil pH", 3.5, 10.0, 6.5, 0.1, help="pH level of soil")

    # Fix: Create columns for environmental parameters
    st.subheader("🌡️ Environmental Parameters")
    temperature = st.slider("Temperature (°C)", 8.0, 45.0, 20.87, 0.01)
    humidity = st.slider("Humidity (%)", 14.0, 100.0, 82.0, 0.1)
    rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 202.9, 0.1)

    if st.button("🔍 Predict Crop"):
        try:
            crop, confidence = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
            model_info = get_model_info()
            
            # Enhanced prediction insights
            st.write("### 📊 Prediction Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                st.success(f"### 🎯 Recommended Crop: {crop}")
                st.info(f"### 📈 Prediction Confidence: {confidence:.2%}")
                st.write(f"Model Type: {model_info['performance']['model_type']}")
                
            with col2:
                # Parameter optimization suggestions
                st.write("#### Parameter Optimization")
                for param, ranges in model_info['recommendations'].items():
                    current_value = locals().get(param.lower())
                    if current_value:
                        if current_value < ranges['low']:
                            st.warning(f"{param} is too low. Increase to {ranges['optimal']}")
                        elif current_value > ranges['high']:
                            st.warning(f"{param} is too high. Decrease to {ranges['optimal']}")
                        else:
                            st.success(f"{param} is within optimal range")

            # Save prediction to history
            prediction_record = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'crop': crop,
                'confidence': confidence,
                'parameters': {
                    'N': N, 'P': P, 'K': K,
                    'temperature': temperature,
                    'humidity': humidity,
                    'ph': ph,
                    'rainfall': rainfall
                }
            }
            st.session_state.prediction_history.append(prediction_record)
            
            # Display result with enhanced styling
            st.success(f"### 🎯 Recommended Crop: {crop}")
            st.info(f"### 📊 Prediction Confidence: {confidence:.2%}")
            
            # Create radar chart for parameters
            parameters = ['N', 'P', 'K', 'pH', 'Temperature', 'Humidity', 'Rainfall']
            values = [N/140, P/145, K/205, ph/10, temperature/45, humidity/100, rainfall/300]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=parameters,
                fill='toself'
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                title="Parameter Distribution"
            )
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")

with tab2:
    st.subheader("📈 Parameter Analysis")
    if len(st.session_state.prediction_history) > 0:
        history_df = pd.DataFrame([x['parameters'] for x in st.session_state.prediction_history])
        
        # Correlation matrix
        corr = history_df.corr()
        fig_corr = px.imshow(corr, title="Parameter Correlation Matrix")
        st.plotly_chart(fig_corr)
        
        # Parameter distributions
        for param in history_df.columns:
            fig_hist = px.histogram(history_df, x=param, title=f"{param} Distribution")
            st.plotly_chart(fig_hist)
    else:
        st.info("Make some predictions to see the analysis!")

with tab3:
    st.subheader("📜 Prediction History")
    if len(st.session_state.prediction_history) > 0:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df)
        
        if st.button("💾 Export History"):
            history_df.to_csv("prediction_history.csv", index=False)
            st.success("History exported to prediction_history.csv")
    else:
        st.info("No prediction history available yet.")

with tab4:
    st.subheader("🤖 AI-Powered Insights")
    
    if len(st.session_state.prediction_history) > 0:
        st.write("### 📈 Crop Success Rate Analysis")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        crop_counts = history_df['crop'].value_counts()
        
        fig_pie = px.pie(values=crop_counts.values, 
                        names=crop_counts.index, 
                        title="Recommended Crops Distribution")
        st.plotly_chart(fig_pie)
        
        # Optimal Conditions Analysis
        st.write("### 🎯 Optimal Growing Conditions")
        for crop in crop_counts.index:
            crop_data = history_df[history_df['crop'] == crop]
            st.write(f"**{crop}** (Average Confidence: {crop_data['confidence'].mean():.2%})")
            
            # Extract parameters for this crop
            params_df = pd.DataFrame([x['parameters'] for x in 
                                    [p for p in st.session_state.prediction_history 
                                     if p['crop'] == crop]])
            
            # Calculate optimal ranges
            ranges = {}
            for param in params_df.columns:
                ranges[param] = {
                    'min': params_df[param].min(),
                    'max': params_df[param].max(),
                    'mean': params_df[param].mean()
                }
            
            # Display optimal ranges
            col1, col2 = st.columns(2)
            with col1:
                st.write("Optimal Ranges:")
                for param, values in ranges.items():
                    st.write(f"- {param}: {values['min']:.2f} - {values['max']:.2f}")
            
            with col2:
                # Create radar chart for optimal values
                parameters = list(ranges.keys())
                values = [ranges[p]['mean'] for p in parameters]
                
                fig = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=parameters,
                    fill='toself'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    showlegend=False,
                    title=f"Optimal Parameters for {crop}"
                )
                st.plotly_chart(fig)
        
        # Seasonal Analysis
        st.write("### 🌺 Seasonal Recommendations")
        months = pd.to_datetime(history_df['timestamp']).dt.month
        season_map = {
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        }
        history_df['season'] = months.map(season_map)
        
        seasonal_crops = pd.crosstab(history_df['season'], history_df['crop'])
        fig_seasonal = px.bar(seasonal_crops, 
                            title="Seasonal Crop Distribution",
                            labels={'value': 'Number of Recommendations'})
        st.plotly_chart(fig_seasonal)
        
        # AI Recommendations
        st.write("### 🤖 AI Recommendations")
        st.info("""
        Based on the analysis of your prediction history:
        
        1. **Best Performing Crops**: The crops with highest confidence scores
        2. **Optimal Conditions**: Detailed parameter ranges for each crop
        3. **Seasonal Patterns**: Which crops are recommended more in certain seasons
        4. **Parameter Importance**: How different parameters affect crop selection
        """)
        
        # Advanced AI Analysis
        st.write("### 🧠 Advanced AI Insights")
        
        # Crop Performance Analysis
        st.write("#### 📊 Crop Performance Trends")
        performance_df = pd.DataFrame({
            'timestamp': pd.to_datetime(history_df['timestamp']),
            'confidence': history_df['confidence'],
            'crop': history_df['crop']
        }).sort_values('timestamp')
        
        fig_trend = px.line(performance_df, 
                           x='timestamp', 
                           y='confidence',
                           color='crop',
                           title="Crop Prediction Confidence Over Time")
        st.plotly_chart(fig_trend)
        
        # Parameter Impact Analysis
        st.write("#### 🎯 Parameter Impact Analysis")
        impact_scores = {}
        for param in params_df.columns:
            correlation = abs(params_df[param].corr(history_df['confidence']))
            impact_scores[param] = correlation
        
        impact_df = pd.DataFrame(list(impact_scores.items()), 
                               columns=['Parameter', 'Impact Score'])
        impact_df = impact_df.sort_values('Impact Score', ascending=False)
        
        fig_impact = px.bar(impact_df, 
                           x='Parameter', 
                           y='Impact Score',
                           title="Parameter Impact on Prediction Confidence")
        st.plotly_chart(fig_impact)
        
        # Anomaly Detection for Growing Conditions
        params_df = pd.DataFrame([x['parameters'] for x in st.session_state.prediction_history])
        scaler = StandardScaler()
        scaled_params = scaler.fit_transform(params_df)
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(scaled_params)
        
        # Visualize unusual conditions
        fig_anomaly = px.scatter(
            params_df,
            x='temperature',
            y='rainfall',
            color=anomalies,
            title="Unusual Growing Conditions Detection",
            labels={'color': 'Normal(-1) vs Unusual(1)'},
            color_discrete_map={-1: 'green', 1: 'red'}
        )
        st.plotly_chart(fig_anomaly)
        
        # Add explanation for unusual conditions
        unusual_count = sum(anomalies == 1)
        if unusual_count > 0:
            st.warning(f"Detected {unusual_count} unusual growing condition(s). Red points indicate conditions that deviate significantly from typical patterns.")
        else:
            st.success("All growing conditions follow typical patterns.")
        
        # Cluster Analysis
        scaler = StandardScaler()
        scaled_params = scaler.fit_transform(params_df)
        kmeans = KMeans(n_clusters=min(3, len(params_df)), random_state=42)
        clusters = kmeans.fit_predict(scaled_params)
        
        # Visualize clusters
        fig_3d = px.scatter_3d(
            params_df,
            x='temperature',
            y='humidity',
            z='rainfall',
            color=clusters,
            title="Parameter Clusters"
        )
        st.plotly_chart(fig_3d)
        
        # Crop Success Prediction
        st.write("### 🎯 Success Rate Prediction")
        success_threshold = 0.7  # 70% confidence
        success_rate = (history_df['confidence'] > success_threshold).mean()
        
        st.metric(
            "Overall Success Rate",
            f"{success_rate:.2%}",
            f"{(success_rate - 0.5):.2%}"
        )
        
        # Crop Recommendations by Season
        st.write("### 🌱 Smart Seasonal Recommendations")
        current_month = datetime.now().month
        current_season = season_map[current_month]
        
        seasonal_success = history_df[history_df['season'] == current_season]
        if not seasonal_success.empty:
            best_crops = seasonal_success.groupby('crop')['confidence'].mean().sort_values(ascending=False)
            
            st.write(f"Top Recommendations for {current_season}:")
            for crop, conf in best_crops.items():
                st.write(f"- {crop}: {conf:.2%} confidence")
        
        # Parameter Optimization
        st.write("### ⚙️ Parameter Optimization")
        # In tab4, update the selectbox key
        selected_crop = st.selectbox("Select Crop for Optimization", crop_counts.index, key='optimization_crop')
        
        if selected_crop:
            crop_params = params_df[history_df['crop'] == selected_crop]
            optimal_params = crop_params.mean()
            
            st.write("Recommended Parameter Adjustments:")
            current_params = pd.Series({
                'N': N, 'P': P, 'K': K,
                'temperature': temperature,
                'humidity': humidity,
                'ph': ph,
                'rainfall': rainfall
            })
            
            for param, optimal in optimal_params.items():
                current = current_params[param]
                diff = optimal - current
                arrow = "↑" if diff > 0 else "↓" if diff < 0 else "✓"
                st.write(f"- {param}: {arrow} ({diff:+.2f})")

with tab5:
    st.subheader("ℹ️ Help & Information")
    st.markdown("""
    ### How to Use
    1. Adjust the sliders for soil and environmental parameters
    2. Click 'Predict Crop' to get recommendations
    3. View detailed analysis in the Analysis tab
    4. Track your predictions in the History tab
    
    ### Parameter Guidelines
    - **Nitrogen (N)**: Essential for leaf growth
    - **Phosphorus (P)**: Important for root development
    - **Potassium (K)**: Helps in overall plant health
    - **pH**: Affects nutrient availability
    - **Temperature**: Influences growth rate
    - **Humidity**: Affects water requirements
    - **Rainfall**: Determines water availability
    """)

# Enhanced sidebar
# Add this after the imports
def get_model_info():
    return {
        'feature_importance': {
            'Rainfall': {'value': 18.36, 'ci_lower': 17.2, 'ci_upper': 19.5},
            'Humidity': {'value': 17.89, 'ci_lower': 16.8, 'ci_upper': 19.0},
            'Nitrogen': {'value': 16.45, 'ci_lower': 15.3, 'ci_upper': 17.6},
            'pH': {'value': 15.51, 'ci_lower': 14.4, 'ci_upper': 16.6},
            'Potassium': {'value': 14.11, 'ci_lower': 13.0, 'ci_upper': 15.2},
            'Temperature': {'value': 9.73, 'ci_lower': 8.6, 'ci_upper': 10.9},
            'Phosphorus': {'value': 7.95, 'ci_lower': 6.8, 'ci_upper': 9.1}
        },
        'performance': {
            'accuracy': 0.90,
            'cv_score': 0.885,
            'precision': 0.88,
            'recall': 0.87,
            'f1_score': 0.89,
            'model_type': 'Random Forest Classifier',
            'last_updated': '2024-01-20',
            'training_samples': 10000,
            'validation_method': '5-fold cross validation'
        },
        'recommendations': {
            'Rainfall': {'low': 50, 'optimal': 200, 'high': 300},
            'Humidity': {'low': 30, 'optimal': 70, 'high': 90},
            'Temperature': {'low': 10, 'optimal': 25, 'high': 35},
            'pH': {'low': 5.5, 'optimal': 6.5, 'high': 7.5},
            'Nitrogen': {'low': 40, 'optimal': 90, 'high': 120},
            'Phosphorus': {'low': 20, 'optimal': 42, 'high': 80},
            'Potassium': {'low': 20, 'optimal': 43, 'high': 100}
        },
        'model_version': '2.0',
        'update_frequency': 'Monthly'
    }

# Update the sidebar section
st.sidebar.title("🔍 Model Information")
model_info = get_model_info()

# Create an expandable section for model details
with st.sidebar.expander("📊 Model Performance Metrics", expanded=True):
    st.metric("Model Accuracy", f"{model_info['performance']['accuracy']*100:.1f}%")
    st.metric("Cross-validation Score", f"{model_info['performance']['cv_score']*100:.1f}%")

# Visualize feature importance
with st.sidebar.expander("🎯 Feature Importance", expanded=True):
    importance_df = pd.DataFrame({
        'Feature': model_info['feature_importance'].keys(),
        'Importance': model_info['feature_importance'].values()
    })
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Add export/import functionality
st.sidebar.subheader("💾 Data Management")
if st.sidebar.button("Export All Data"):
    data = {
        'history': st.session_state.prediction_history,
        'model_info': {
            'accuracy': 0.90,
            'cv_score': 0.885
        }
    }
    with open('crop_prediction_data.json', 'w') as f:
        json.dump(data, f)
    st.sidebar.success("Data exported successfully!")

uploaded_file = st.sidebar.file_uploader("Import Previous Data", type=['json'])
if uploaded_file is not None:
    data = json.load(uploaded_file)
    st.session_state.prediction_history = data['history']
    st.sidebar.success("Data imported successfully!")

# Add these new styles after the existing CSS
st.markdown("""
    <style>
    /* Shimmering effect for cards */
    .feature-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent,
            rgba(255, 255, 255, 0.1),
            transparent
        );
        transform: rotate(30deg);
        animation: shimmer 6s linear infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) rotate(30deg); }
        100% { transform: translateX(100%) rotate(30deg); }
    }
    
    /* Enhanced button hover state */
    .stButton > button:active {
        transform: scale(0.95) translateY(-8px);
    }
    
    /* Pulsing effect for important metrics */
    .stMetric:hover {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Smooth transitions for all interactive elements */
    .stSelectbox, .stSlider, .stTextInput {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Focus states */
    .stSelectbox:focus-within, .stSlider:focus-within, .stTextInput:focus-within {
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Add this after the Help tab content
with tab6:
    st.subheader("🗓️ Crop Calendar")
    
    # Near the top of the file
    from dotenv import load_dotenv
    load_dotenv()
    
    # Replace the API key section
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        st.error("OpenWeather API key not found in environment variables")
        st.stop()
    
    crop_calendar = {
        'rice': {'planting': ['Mar', 'Jul'], 'harvesting': ['Jun', 'Nov']},
        'wheat': {'planting': ['Oct', 'Nov'], 'harvesting': ['Apr', 'May']},
        'maize': {'planting': ['Jun', 'Jul'], 'harvesting': ['Sep', 'Oct']},
        'cotton': {'planting': ['Mar', 'Apr'], 'harvesting': ['Sep', 'Oct']},
        'sugarcane': {'planting': ['Jan', 'Feb'], 'harvesting': ['Dec', 'Jan']},
        'potato': {'planting': ['Sep', 'Oct'], 'harvesting': ['Jan', 'Feb']},
        'tomato': {'planting': ['Feb', 'Mar'], 'harvesting': ['May', 'Jun']},
        'onion': {'planting': ['Oct', 'Nov'], 'harvesting': ['Mar', 'Apr']},
        'garlic': {'planting': ['Sep', 'Oct'], 'harvesting': ['Feb', 'Mar']},
        'peas': {'planting': ['Sep', 'Nov'], 'harvesting': ['Jan', 'Mar']},
        'beans': {'planting': ['Mar', 'Apr'], 'harvesting': ['Jun', 'Jul']},
        'cucumber': {'planting': ['Feb', 'Mar'], 'harvesting': ['Apr', 'May']},
        'carrot': {'planting': ['Sep', 'Oct'], 'harvesting': ['Dec', 'Jan']},
        'cabbage': {'planting': ['Sep', 'Oct'], 'harvesting': ['Dec', 'Jan']},
        'cauliflower': {'planting': ['Sep', 'Oct'], 'harvesting': ['Dec', 'Jan']},
        'broccoli': {'planting': ['Aug', 'Sep'], 'harvesting': ['Nov', 'Dec']},
        'spinach': {'planting': ['Sep', 'Oct'], 'harvesting': ['Nov', 'Dec']},
        'lettuce': {'planting': ['Sep', 'Oct'], 'harvesting': ['Nov', 'Dec']},
        'bell_pepper': {'planting': ['Feb', 'Mar'], 'harvesting': ['May', 'Jun']},
        'eggplant': {'planting': ['Feb', 'Mar'], 'harvesting': ['May', 'Jun']},
        'okra': {'planting': ['Mar', 'Apr'], 'harvesting': ['Jun', 'Jul']},
        'pumpkin': {'planting': ['Jun', 'Jul'], 'harvesting': ['Oct', 'Nov']},
        'watermelon': {'planting': ['Feb', 'Mar'], 'harvesting': ['May', 'Jun']},
        'muskmelon': {'planting': ['Feb', 'Mar'], 'harvesting': ['May', 'Jun']},
        'groundnut': {'planting': ['Jun', 'Jul'], 'harvesting': ['Oct', 'Nov']},
        'sunflower': {'planting': ['Jun', 'Jul'], 'harvesting': ['Sep', 'Oct']},
        'soybean': {'planting': ['Jun', 'Jul'], 'harvesting': ['Sep', 'Oct']},
        'mustard': {'planting': ['Oct', 'Nov'], 'harvesting': ['Feb', 'Mar']},
        'coconut': {'planting': ['Jun', 'Jul'], 'harvesting': ['Year-round']},
        'mango': {'planting': ['Jul', 'Aug'], 'harvesting': ['Apr', 'May']}
    }
    
    # Group crops by category
    crop_categories = {
        'Cereals': ['rice', 'wheat', 'maize'],
        'Commercial Crops': ['cotton', 'sugarcane', 'sunflower', 'coconut'],
        'Vegetables': ['potato', 'tomato', 'onion', 'garlic', 'peas', 'beans', 'cucumber', 'carrot', 'cabbage', 'cauliflower', 'broccoli', 'spinach', 'lettuce', 'bell_pepper', 'eggplant', 'okra', 'pumpkin'],
        'Fruits': ['watermelon', 'muskmelon', 'mango'],
        'Oilseeds': ['groundnut', 'soybean', 'mustard']
    }
    
    # Define growing features for all crops
    growing_features = {
        'rice': {
            'soil_type': 'Clay or clay loam soil',
            'water_needs': 'High (150-300 cm per season)',
            'sunlight': 'Full sun (6-8 hours daily)',
            'spacing': '20-25 cm between plants',
            'fertilizer': 'NPK ratio 120:60:60 kg/ha',
            'pests': ['Stem borers', 'Rice bugs', 'Leaf hoppers'],
            'diseases': ['Blast', 'Bacterial blight'],
            'companion_plants': ['Azolla', 'Duckweed'],
            'tips': "Requires standing water and warm temperatures (25-35°C). Maintain water depth of 2-5cm during growth."
        }
    }
    
    # Add default features for other crops
    for crop in crop_calendar.keys():
        if crop not in growing_features:
            growing_features[crop] = {
                'soil_type': 'Well-draining soil',
                'water_needs': 'Moderate',
                'sunlight': 'Full sun',
                'spacing': '30-45 cm between plants',
                'fertilizer': 'Balanced NPK',
                'pests': ['Common pests'],
                'diseases': ['Common diseases'],
                'companion_plants': ['Compatible plants'],
                'tips': "General growing tips for this crop"
            }
    
    # Category selection
    category = st.selectbox("Select Crop Category", list(crop_categories.keys()))
    selected_crop = st.selectbox("Select Crop", crop_categories[category])
    
    if selected_crop:
        st.write(f"### {selected_crop.title()} Growing Calendar")
        
        # Create a modern card layout for crop features
        # Updated feature card styling
        st.markdown("""
            <style>
            .feature-card {
                background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
                padding: 25px;
                border-radius: 20px;
                border: 1px solid rgba(76, 175, 80, 0.2);
                margin: 15px 0;
                box-shadow: 0 8px 32px rgba(76, 175, 80, 0.1);
                backdrop-filter: blur(8px);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .feature-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 12px 40px rgba(76, 175, 80, 0.15);
            }
            .feature-title {
                color: #2E7D32;
                font-size: 1.3em;
                font-weight: 600;
                margin-bottom: 20px;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Basic Requirements Card
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="feature-title">🌱 Basic Requirements</p>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Soil Type", growing_features[selected_crop]['soil_type'])
        with col2:
            st.metric("Water Needs", growing_features[selected_crop]['water_needs'])
        with col3:
            st.metric("Sunlight", growing_features[selected_crop]['sunlight'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Technical Details Card
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="feature-title">🔧 Technical Details</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Plant Spacing", growing_features[selected_crop]['spacing'])
        with col2:
            st.metric("Fertilizer", growing_features[selected_crop]['fertilizer'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Common Issues Card
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<p class="feature-title">⚠️ Common Issues</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Pests:**")
            for pest in growing_features[selected_crop]['pests']:
                st.write(f"• {pest}")
        with col2:
            st.write("**Diseases:**")
            for disease in growing_features[selected_crop]['diseases']:
                st.write(f"• {disease}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tips and Companion Plants in Expandable Cards
        col1, col2 = st.columns(2)
        with col1:
            with st.expander("📝 Growing Tips", expanded=True):
                st.info(growing_features[selected_crop]['tips'])
        with col2:
            with st.expander("🌿 Companion Plants", expanded=True):
                for plant in growing_features[selected_crop]['companion_plants']:
                    st.success(f"• {plant}")
        
        # Add visual progress indicator
        growth_stages = {
            'Seedling': 20,
            'Vegetative': 40,
            'Flowering': 60,
            'Maturity': 80,
            'Harvest': 100
        }
        
        st.write("### 📈 Growth Stage Timeline")
        for stage, progress in growth_stages.items():
            st.write(f"**{stage}**")
            st.progress(progress/100)
    else:
        st.info(growing_tips.get(selected_crop, "Detailed features for this crop will be added soon!"))

def validate_parameters(N, P, K, temperature, humidity, ph, rainfall):
    validation_rules = {
        'N': {'min': 0, 'max': 140},
        'P': {'min': 5, 'max': 145},
        'K': {'min': 5, 'max': 205},
        'temperature': {'min': 8.0, 'max': 45.0},
        'humidity': {'min': 14.0, 'max': 100.0},
        'ph': {'min': 3.5, 'max': 10.0},
        'rainfall': {'min': 20.0, 'max': 300.0}
    }
    
    params = locals()
    for param, rules in validation_rules.items():
        value = params[param]
        if not rules['min'] <= value <= rules['max']:
            raise ValueError(f"{param} must be between {rules['min']} and {rules['max']}")