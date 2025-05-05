def export_data(N, P, K, fertilizer_tips):
    """Export fertilizer analysis data to CSV and JSON formats"""
    try:
        # Create analysis data dictionary
        analysis_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'soil_parameters': {
                'Nitrogen': N,
                'Phosphorus': P,
                'Potassium': K
            },
            'recommendations': fertilizer_tips
        }
        
        # Save as JSON
        json_file = 'fertilizer_analysis.json'
        try:
            with open(json_file, 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = []
        
        existing_data.append(analysis_data)
        with open(json_file, 'w') as f:
            json.dump(existing_data, f, indent=4)
        
        # Save as CSV
        df = pd.DataFrame({
            'Date': [analysis_data['timestamp']],
            'Nitrogen': [N],
            'Phosphorus': [P],
            'Potassium': [K],
            'Recommendations': [str(fertilizer_tips)]
        })
        
        csv_file = 'fertilizer_analysis.csv'
        if not os.path.exists(csv_file):
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(csv_file, mode='a', header=False, index=False)
        
        st.success("✅ Data exported successfully!")
        return True
        
    except Exception as e:
        st.error(f"❌ Error exporting data: {str(e)}")
        return False

# Usage in your main app
import os
import json
import pandas as pd
from datetime import datetime

# Get recommendations
fertilizer_tips = get_fertilizer_recommendations(N, P, K)

# Display analysis
display_fertilizer_analysis(N, P, K)

# Export data
if st.button("Export Analysis Data"):
    export_data(N, P, K, fertilizer_tips)

# Show exported data
if st.checkbox("Show Exported Data"):
    try:
        df = pd.read_csv('fertilizer_analysis.csv')
        st.write("📊 Exported Data History")
        st.dataframe(df)
    except FileNotFoundError:
        st.info("No exported data found yet.")