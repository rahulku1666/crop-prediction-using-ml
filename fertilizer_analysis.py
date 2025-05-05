def display_fertilizer_analysis(N, P, K):
    """Display comprehensive fertilizer analysis and recommendations"""
    
    # Get fertilizer recommendations
    fertilizer_tips = get_fertilizer_recommendations(N, P, K)
    
    # Display NPK Status
    st.subheader("📊 Current NPK Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nitrogen (N)", f"{N} mg/kg", delta="Low" if N < 50 else "Optimal")
    with col2:
        st.metric("Phosphorus (P)", f"{P} mg/kg", delta="Low" if P < 20 else "Optimal")
    with col3:
        st.metric("Potassium (K)", f"{K} mg/kg", delta="Low" if K < 30 else "Optimal")
    
    # Display detailed recommendations
    st.subheader("🌿 Fertilizer Recommendations")
    for tip in fertilizer_tips:
        with st.expander(f"💡 {tip['nutrient']} Recommendations"):
            st.info(f"""
            {tip['message']}
            📝 Application: {tip['application']}
            ⏰ Timing: {tip['timing']}
            ⚠️ Precaution: {tip['precaution']}
            """)
            
            # Display alternatives in columns
            st.write("🔄 Alternative Options:")
            alt_col1, alt_col2 = st.columns(2)
            with alt_col1:
                st.write("Chemical Alternatives:")
                for alt in tip['alternatives']:
                    st.write(f"- {alt}")
            with alt_col2:
                st.write("Organic Options:")
                for org in tip['organic_options']:
                    st.write(f"- {org}")
    
    # Historical Data Analysis
    if st.checkbox("📈 Show Historical NPK Trends"):
        show_historical_trends()

def show_historical_trends():
    """Display historical NPK trends from saved predictions"""
    try:
        with open("prediction_history.json", "r") as f:
            history = json.load(f)
        
        if history:
            df = pd.DataFrame(history)
            st.line_chart(df[['N', 'P', 'K']])
            
            # Show statistics
            st.write("📊 Statistical Summary")
            st.dataframe(df[['N', 'P', 'K']].describe())
        else:
            st.info("No historical data available yet.")
    except Exception as e:
        st.error(f"Error loading historical data: {e}")

def save_fertilizer_analysis(N, P, K, recommendations):
    """Save fertilizer analysis results"""
    analysis_data = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'npk_values': {'N': N, 'P': P, 'K': K},
        'recommendations': recommendations
    }
    
    try:
        with open('fertilizer_analysis_history.json', 'a') as f:
            json.dump(analysis_data, f)
            f.write('\n')
        return True
    except Exception as e:
        print(f"Error saving analysis: {e}")
        return False

# Usage
display_fertilizer_analysis(N, P, K)