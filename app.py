import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from test_model import predict_crop
import time
from datetime import datetime
import hashlib
import pymongo

# MongoDB connection
def connect_to_mongodb():
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["farm_app"]
        return db
    except Exception as e:
        st.error(f"Could not connect to MongoDB: {e}")
        return None

# User collection operations
def create_user(db, username, password, email, name, location):
    users = db["users"]
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    user = {
        "username": username,
        "password": hashed_password,
        "email": email,
        "name": name,
        "location": location,
        "created_at": datetime.now(),
        "last_login": None
    }
    try:
        users.insert_one(user)
        return True
    except Exception as e:
        st.error(f"Error creating user: {e}")
        return False

def verify_user(db, username, password):
    users = db["users"]
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    user = users.find_one({"username": username, "password": hashed_password})
    if user:
        users.update_one(
            {"username": username},
            {"$set": {"last_login": datetime.now()}}
        )
        return True
    return False

def show_login_page():
    st.title("🌾 Smart Crop Recommendation System")
    
    # Create tabs for login and registration
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            st.markdown("""
                <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h2 style='text-align: center; color: #2E7D32;'>👤 Farmer Login</h2>
                </div>
            """, unsafe_allow_html=True)
            
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                submit_login = st.form_submit_button("Login", use_container_width=True)
                
                if submit_login:
                    db = connect_to_mongodb()
                    if db and verify_user(db, username, password):
                        st.success("Login successful!")
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
    
    with tab2:
        with st.form("register_form"):
            st.markdown("""
                <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h2 style='text-align: center; color: #2E7D32;'>📝 New Farmer Registration</h2>
                </div>
            """, unsafe_allow_html=True)
            
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            email = st.text_input("Email")
            name = st.text_input("Full Name")
            location = st.text_input("Location")
            
            submit_register = st.form_submit_button("Register", use_container_width=True)
            
            if submit_register:
                if new_password != confirm_password:
                    st.error("Passwords do not match!")
                elif not all([new_username, new_password, email, name, location]):
                    st.error("All fields are required!")
                else:
                    db = connect_to_mongodb()
                    if db and create_user(db, new_username, new_password, email, name, location):
                        st.success("Registration successful! Please login.")
                        time.sleep(2)
                        st.rerun()

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''

def main():
    # Page configuration
    st.set_page_config(
        page_title="Smart Crop Recommendation System",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Your existing CSS styles here
    st.markdown("""
        <style>
        .main {
            background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
            padding: 2rem;
        }
        .stButton>button {
            background: linear-gradient(45deg, #2E7D32, #43A047);
            color: white;
            width: 100%;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
            border: none;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin: 10px 0;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #43A047, #2E7D32);
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .stMetric {
            background: linear-gradient(135deg, #ffffff, #f8f9fa);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            border: 1px solid #e0e0e0;
            margin: 10px 0;
        }
        .stMetric:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        .stSelectbox {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 5px;
            border: 1px solid #e0e0e0;
            margin: 10px 0;
        }
        h1 {
            color: #1B5E20;
            text-align: center;
            padding: 1rem;
            border-bottom: 2px solid #4CAF50;
            margin-bottom: 2rem;
            font-size: 2.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .stSubheader {
            color: #2E7D32;
            font-weight: bold;
            border-left: 4px solid #4CAF50;
            padding-left: 10px;
            margin: 20px 0;
        }
        .stProgress > div > div {
            background-color: #4CAF50;
        }
        .stTab {
            background-color: #E8F5E9;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .css-1v0mbdj.etr89bj1 {
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Enhanced Sidebar with better styling and more features
    with st.sidebar:
        st.image("https://img.freepik.com/free-vector/farm-logo_23-2147503611.jpg", width=200)
        st.title("🌿 Smart Farming")
        
        # Enhanced user profile section
        with st.expander("👤 Farmer Profile", expanded=True):
            st.write(f"Welcome, {st.session_state.username}!")
            st.progress(100)
            st.text(f"Last Login: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            
            # Add farmer details
            farmer_details = {
                'farmer1': {
                    'name': 'John Smith',
                    'location': 'Karnataka',
                    'farm_size': '5 acres',
                    'farming_type': 'Organic'
                },
                'farmer2': {
                    'name': 'Mary Johnson',
                    'location': 'Maharashtra',
                    'farm_size': '8 acres',
                    'farming_type': 'Traditional'
                }
            }
            
            if st.session_state.username in farmer_details:
                details = farmer_details[st.session_state.username]
                st.write("**Farmer Details:**")
                for key, value in details.items():
                    st.text(f"{key.replace('_', ' ').title()}: {value}")
            
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = ''
                st.rerun()  # Updated from experimental_rerun()

        # Weather widget in sidebar
        with st.expander("🌤️ Weather Updates", expanded=True):
            st.metric("Temperature", "25°C", "1.2°C")
            st.metric("Humidity", "65%", "-5%")
            st.metric("Rainfall Chance", "30%", "10%")
        
        # Enhanced navigation
        page = st.radio("📍 Navigation",
            ["🎯 Prediction", "📊 Analysis", "📚 Knowledge Base", "🌱 Crop Calendar", "❓ Help", "📱 Contact"],
            help="Choose a section to navigate"
        )

    # Page routing with enhanced content
    if page == "🎯 Prediction":
        show_prediction_page()
    elif page == "📊 Analysis":
        show_analysis_page()
    elif page == "📚 Knowledge Base":
        show_knowledge_base()
    elif page == "🌱 Crop Calendar":
        show_crop_calendar()
    elif page == "📱 Contact":
        show_contact_page()
    else:
        show_help_page()

def show_knowledge_base():
    st.title("📚 Agricultural Knowledge Base")
    
    tab1, tab2, tab3 = st.tabs(["🌱 Crop Guide", "🌡️ Climate Impact", "💧 Water Management"])
    
    with tab1:
        st.subheader("Comprehensive Crop Guide")
        crop_info = {
            "Rice": {
                "season": "Kharif",
                "water_req": "High",
                "soil_type": "Clay",
                "duration": "120-150 days",
                "fertilizer": "NPK 120:60:60"
            },
            "Wheat": {
                "season": "Rabi",
                "water_req": "Medium",
                "soil_type": "Loamy",
                "duration": "120-150 days",
                "fertilizer": "NPK 120:60:40"
            }
        }
        selected_crop = st.selectbox("Select Crop", list(crop_info.keys()))
        if selected_crop:
            st.json(crop_info[selected_crop])
            
    with tab2:
        st.subheader("Climate Impact Analysis")
        climate_data = pd.DataFrame({
            'Temperature': ['Low', 'Medium', 'High'],
            'Impact': [-20, 0, -30]
        })
        fig = px.bar(climate_data, x='Temperature', y='Impact',
                    title='Temperature Impact on Crop Yield')
        st.plotly_chart(fig)

def show_crop_calendar():
    st.title("🌱 Seasonal Crop Calendar")
    
    seasons = {
        "Kharif": ["Rice", "Maize", "Soybean", "Cotton"],
        "Rabi": ["Wheat", "Barley", "Peas", "Mustard"],
        "Zaid": ["Watermelon", "Muskmelon", "Cucumber"]
    }
    
    current_month = datetime.now().strftime("%B")
    st.subheader(f"Current Month: {current_month}")
    
    for season, crops in seasons.items():
        with st.expander(f"🗓️ {season} Season Crops"):
            st.write(f"**Recommended Crops:** {', '.join(crops)}")
            
            # Add planting schedule
            schedule_df = pd.DataFrame({
                'Crop': crops,
                'Planting Time': ['Early Season'] * len(crops),
                'Harvest Time': ['Late Season'] * len(crops)
            })
            st.table(schedule_df)

def show_prediction_page():
    st.title("🌾 Smart Crop Recommendation System")
    st.write("Enter soil parameters and environmental conditions to get crop recommendations")
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Soil Nutrients")
        N = st.number_input("Nitrogen (N) content", 0, 140, 90)
        P = st.number_input("Phosphorus (P) content", 5, 145, 42)
        K = st.number_input("Potassium (K) content", 5, 205, 43)
    
    with col2:
        st.subheader("🌡️ Environmental Factors")
        temperature = st.slider("Temperature (°C)", 0.0, 45.0, 20.87)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 82.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 202.935)
    
    with col3:
        st.subheader("🧪 Soil Properties")
        ph = st.slider("pH value", 0.0, 14.0, 6.5)
        soil_type = st.selectbox("Soil Type", ["Clay", "Sandy", "Loamy", "Black", "Red", "Alluvial"])
        irrigation = st.selectbox("Irrigation Method", ["Drip", "Sprinkler", "Flood", "Furrow", "Manual"])

    # Add predict button with spinner
    if st.button("🔍 Predict Suitable Crop", use_container_width=True):
        with st.spinner('Analyzing soil parameters...'):
            try:
                # Make prediction
                crop, confidence = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
                
                # Display results
                st.success("### 🎯 Prediction Results")
                
                # Results in columns
                col4, col5 = st.columns(2)
                with col4:
                    st.metric("Recommended Crop", crop)
                    st.metric("Soil Health", "Good" if ph > 6.0 and ph < 7.5 else "Needs Attention")
                with col5:
                    st.metric("Prediction Confidence", f"{confidence:.2%}")
                    
                # Display recommendations
                st.info("### 🌱 Key Recommendations")
                st.write(f"""
                - Irrigation Schedule: {get_irrigation_schedule(crop)}
                - Best Planting Time: {get_planting_season(crop)}
                """)
                
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")

def show_analysis_page():
    st.title("📈 Crop Analysis Dashboard")
    
    # Add tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["Yield Analysis", "Cost Analysis", "Market Trends"])
    
    with tab1:
        st.subheader("📊 Crop Yield Trends")
        yield_data = pd.DataFrame({
            'Year': [2018, 2019, 2020, 2021, 2022],
            'Yield': [75, 82, 85, 89, 92]
        })
        st.line_chart(yield_data.set_index('Year'))

    with tab2:
        st.subheader("💰 Cost Breakdown")
        cost_data = pd.DataFrame({
            'Category': ['Seeds', 'Fertilizers', 'Labor', 'Equipment'],
            'Cost': [2000, 3000, 4000, 1000]
        })
        fig = px.pie(cost_data, values='Cost', names='Category')
        st.plotly_chart(fig)
        
    with tab3:
        st.subheader("📈 Market Price Trends")
        st.info("Coming soon: Real-time market price analysis")

def show_contact_page():
    st.title("📱 Contact & Support")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 📬 Get in Touch")
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")
        if st.button("Send Message"):
            st.success("Message sent successfully!")
    
    with col2:
        st.write("### 📞 Help Desk")
        st.info("""
        **24/7 Support**
        - Email: support@smartfarm.com
        - Phone: +1-234-567-8900
        - Chat: Available 9 AM - 5 PM
        """)

def show_help_page():
    st.title("❓ Help & Guidelines")
    st.write("Learn how to use the system effectively")
    
    with st.expander("📖 How to Use"):
        st.write("""
        1. Enter your soil parameters (N, P, K values)
        2. Input environmental conditions
        3. Select soil type and irrigation method
        4. Click predict to get recommendations
        """)
    
    with st.expander("🎯 Understanding Results"):
        st.write("""
        - Confidence score indicates prediction reliability
        - Soil health is based on pH levels
        - Charts show nutrient balance analysis
        """)

def get_irrigation_schedule(crop):
    # Placeholder function - replace with actual implementation
    return "Every 2-3 days depending on weather conditions"

def get_soil_recommendations(soil_type):
    # Placeholder function - replace with actual implementation
    return "Regular organic matter addition and proper drainage maintenance"

def get_planting_season(crop):
    # Placeholder function - replace with actual implementation
    return "Early spring to late summer"

if __name__ == "__main__":
    main()