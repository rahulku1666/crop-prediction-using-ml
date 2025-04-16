import pandas as pd
import numpy as np

# Create sample data with better correlations
np.random.seed(42)
n_samples = 2200

# Define crop requirements (approximate ranges for each crop)
crop_requirements = {
    'rice': {'N': (60, 120), 'P': (30, 90), 'K': (30, 90), 'temp': (20, 35), 'humidity': (80, 95), 'ph': (5.5, 6.5), 'rainfall': (200, 300)},
    'wheat': {'N': (100, 140), 'P': (50, 100), 'K': (40, 80), 'temp': (15, 25), 'humidity': (40, 70), 'ph': (6.0, 7.0), 'rainfall': (60, 100)},
    'maize': {'N': (80, 130), 'P': (40, 80), 'K': (30, 60), 'temp': (20, 30), 'humidity': (50, 80), 'ph': (5.5, 7.5), 'rainfall': (80, 200)},
    'chickpea': {'N': (40, 80), 'P': (40, 60), 'K': (20, 40), 'temp': (15, 30), 'humidity': (40, 60), 'ph': (6.0, 8.0), 'rainfall': (40, 100)},
    'cotton': {'N': (100, 140), 'P': (40, 80), 'K': (40, 80), 'temp': (25, 35), 'humidity': (50, 80), 'ph': (5.5, 8.0), 'rainfall': (80, 200)},
    'banana': {'N': (80, 120), 'P': (20, 60), 'K': (40, 100), 'temp': (20, 30), 'humidity': (75, 85), 'ph': (6.0, 7.5), 'rainfall': (160, 300)},
    'mango': {'N': (50, 100), 'P': (25, 75), 'K': (50, 100), 'temp': (24, 35), 'humidity': (65, 75), 'ph': (5.5, 7.5), 'rainfall': (100, 250)},
    'papaya': {'N': (60, 110), 'P': (30, 75), 'K': (45, 85), 'temp': (22, 33), 'humidity': (70, 85), 'ph': (6.0, 7.0), 'rainfall': (120, 250)},
    'coconut': {'N': (70, 120), 'P': (35, 80), 'K': (60, 120), 'temp': (25, 35), 'humidity': (70, 90), 'ph': (5.5, 7.0), 'rainfall': (150, 300)},
    'tea': {'N': (50, 90), 'P': (20, 60), 'K': (30, 70), 'temp': (18, 30), 'humidity': (70, 90), 'ph': (4.5, 5.5), 'rainfall': (200, 300)},
    'cashew': {'N': (40, 80), 'P': (25, 65), 'K': (35, 75), 'temp': (20, 35), 'humidity': (60, 80), 'ph': (5.0, 6.5), 'rainfall': (100, 200)}
}

data = []
for crop, ranges in crop_requirements.items():
    samples_per_crop = n_samples // len(crop_requirements)
    
    for _ in range(samples_per_crop):
        sample = {
            'N': np.random.uniform(ranges['N'][0], ranges['N'][1]),
            'P': np.random.uniform(ranges['P'][0], ranges['P'][1]),
            'K': np.random.uniform(ranges['K'][0], ranges['K'][1]),
            'temperature': np.random.uniform(ranges['temp'][0], ranges['temp'][1]),
            'humidity': np.random.uniform(ranges['humidity'][0], ranges['humidity'][1]),
            'ph': np.random.uniform(ranges['ph'][0], ranges['ph'][1]),
            'rainfall': np.random.uniform(ranges['rainfall'][0], ranges['rainfall'][1]),
            'label': crop
        }
        data.append(sample)

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the data
df.to_csv('Crop_recommendation.csv', index=False)