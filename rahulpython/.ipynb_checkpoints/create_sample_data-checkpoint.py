import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n_samples = 1000

data = {
    'N': np.random.randint(0, 140, n_samples),
    'P': np.random.randint(5, 145, n_samples),
    'K': np.random.randint(5, 205, n_samples),
    'temperature': np.random.uniform(8.83, 43.68, n_samples),
    'humidity': np.random.uniform(14.26, 99.98, n_samples),
    'ph': np.random.uniform(3.50, 9.94, n_samples),
    'rainfall': np.random.uniform(20.21, 298.56, n_samples),
    'crop': np.random.choice(['rice', 'wheat', 'maize', 'cotton'], n_samples)
}

df = pd.DataFrame(data)
df.to_csv('crop_data.csv', index=False)
print("Sample dataset created successfully!")