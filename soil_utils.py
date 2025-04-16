def get_soil_based_crops():
    return {
        "Alluvial": ["Rice", "Wheat", "Sugarcane", "Vegetables"],
        "Black": ["Cotton", "Soybean", "Groundnut", "Pulses"],
        "Red": ["Millet", "Pulses", "Oilseeds", "Groundnut"],
        "Laterite": ["Tea", "Coffee", "Cashew", "Rubber"],
        "Mountain": ["Tea", "Apples", "Spices", "Medicinal Plants"],
        "Desert": ["Barley", "Pearl Millet", "Guar", "Date Palm"]
    }

def get_soil_characteristics():
    return {
        "Alluvial": {
            "pH_range": (6.5, 7.5),
            "N_range": (60, 120),
            "P_range": (30, 60),
            "K_range": (40, 80),
            "description": "Rich in nutrients, good water retention"
        },
        "Black": {
            "pH_range": (6.5, 8.5),
            "N_range": (40, 100),
            "P_range": (20, 50),
            "K_range": (30, 70),
            "description": "High water retention, rich in calcium"
        },
        "Red": {
            "pH_range": (6.0, 7.0),
            "N_range": (30, 90),
            "P_range": (15, 45),
            "K_range": (25, 65),
            "description": "Well-drained, suitable for multiple crops"
        },
        "Laterite": {
            "pH_range": (5.5, 6.5),
            "N_range": (20, 80),
            "P_range": (10, 40),
            "K_range": (20, 60),
            "description": "Acidic, needs proper management"
        },
        "Mountain": {
            "pH_range": (5.0, 6.0),
            "N_range": (30, 70),
            "P_range": (20, 35),
            "K_range": (25, 55),
            "description": "Good organic matter content"
        },
        "Desert": {
            "pH_range": (7.0, 8.5),
            "N_range": (10, 50),
            "P_range": (5, 30),
            "K_range": (15, 45),
            "description": "Low in nutrients, needs enrichment"
        }
    }

def get_soil_info_styles():
    return """
        <style>
        .soil-info {
            padding: 15px;
            border-radius: 10px;
            background: linear-gradient(135deg, rgba(129, 199, 132, 0.1), rgba(76, 175, 80, 0.05));
            margin-bottom: 20px;
        }
        </style>
    """

def get_soil_based_crops():
    return {
        "Alluvial": ["Rice", "Wheat", "Sugarcane"],
        "Black": ["Cotton", "Soybean", "Groundnut"],
        "Red": ["Millet", "Pulses", "Oilseeds"],
        "Laterite": ["Tea", "Coffee", "Cashew"],
        "Mountain": ["Tea", "Apples", "Spices"],
        "Desert": ["Barley", "Pearl Millet", "Guar"]
    }

def display_soil_parameters(st):
    st.markdown(get_soil_info_styles(), unsafe_allow_html=True)
    
    soil_type = st.selectbox("Select Soil Type", ["Alluvial", "Black", "Red", "Laterite", "Mountain", "Desert"])
    soil_based_crops = get_soil_based_crops()
    
    if soil_type:
        st.markdown(f"""
            <div class="soil-info">
                <h4>🌱 Recommended Crops for {soil_type} Soil:</h4>
                <p style='font-size: 1.1em; margin-top: 10px;'>{', '.join(soil_based_crops[soil_type])}</p>
            </div>
        """, unsafe_allow_html=True)

    N = st.slider("Nitrogen (N) mg/kg", 0, 140, 90, help="Amount of Nitrogen in soil")
    P = st.slider("Phosphorus (P) mg/kg", 5, 145, 42, help="Amount of Phosphorus in soil")
    K = st.slider("Potassium (K) mg/kg", 5, 205, 43, help="Amount of Potassium in soil")
    ph = st.slider("Soil pH", 3.5, 10.0, 6.5, 0.1, help="pH level of soil")
    
    return N, P, K, ph, soil_type