import json
import pandas as pd
from datetime import datetime

# Get recommendations and display analysis
display_fertilizer_analysis(N, P, K)

# Save analysis if needed
fertilizer_tips = get_fertilizer_recommendations(N, P, K)
save_fertilizer_analysis(N, P, K, fertilizer_tips)