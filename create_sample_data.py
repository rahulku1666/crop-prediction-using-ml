import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n_samples = 1000

# Define crop-specific conditions
crop_conditions = {
    'rice': {
        'temp_range': (20, 35),
        'humidity_range': (60, 95),
        'ph_range': (5.5, 7.5),
        'rainfall_range': (1000, 3000),
        'N_range': (80, 200),    # Nitrogen requirements for rice
        'P_range': (20, 60),     # Phosphorus requirements for rice
        'K_range': (40, 120)     # Potassium requirements for rice
    },
    'wheat': {
        'temp_range': (15, 25),
        'humidity_range': (40, 70),
        'ph_range': (6.0, 7.5),
        'rainfall_range': (450, 650),
        'N_range': (100, 250),   # Nitrogen requirements for wheat
        'P_range': (50, 100),    # Phosphorus requirements for wheat
        'K_range': (50, 150)     # Potassium requirements for wheat
    },
    'maize': {
        'temp_range': (20, 30),
        'humidity_range': (50, 80),
        'ph_range': (5.5, 7.5),
        'rainfall_range': (500, 800),
        'N_range': (150, 300),   # Nitrogen requirements for maize
        'P_range': (30, 80),     # Phosphorus requirements for maize
        'K_range': (40, 100)     # Potassium requirements for maize
    },
    'cotton': {
        'temp_range': (25, 35),
        'humidity_range': (40, 60),
        'ph_range': (5.5, 8.0),
        'rainfall_range': (600, 1100),
        'N_range': (100, 180),   # Nitrogen requirements for cotton
        'P_range': (40, 90),     # Phosphorus requirements for cotton
        'K_range': (60, 120)     # Potassium requirements for cotton
    }
}

# Generate crop types first
crops = np.random.choice(['rice', 'wheat', 'maize', 'cotton'], n_samples)

# Initialize arrays for all conditions
temperature = np.zeros(n_samples)
humidity = np.zeros(n_samples)
ph = np.zeros(n_samples)
rainfall = np.zeros(n_samples)
N = np.zeros(n_samples)
P = np.zeros(n_samples)
K = np.zeros(n_samples)

# Generate conditions based on crop requirements
for i, crop in enumerate(crops):
    conditions = crop_conditions[crop]
    temperature[i] = np.random.uniform(*conditions['temp_range'])
    humidity[i] = np.random.uniform(*conditions['humidity_range'])
    ph[i] = np.random.uniform(*conditions['ph_range'])
    rainfall[i] = np.random.uniform(*conditions['rainfall_range'])
    N[i] = np.random.uniform(*conditions['N_range'])
    P[i] = np.random.uniform(*conditions['P_range'])
    K[i] = np.random.uniform(*conditions['K_range'])

data = {
    'N': N,                                         # Nitrogen content in soil (kg/ha)
    'P': P,                                         # Phosphorus content in soil (kg/ha)
    'K': K,                                         # Potassium content in soil (kg/ha)
    'temperature': temperature,                      # Temperature in Celsius
    'humidity': humidity,                           # Relative humidity (%)
    'ph': ph,                                       # Soil pH level
    'rainfall': rainfall,                           # Annual rainfall (mm)
    'crop': crops                                   # Target crop
}

df = pd.DataFrame(data)
df.to_csv('crop_data.csv', index=False)
print("Sample dataset created successfully!")