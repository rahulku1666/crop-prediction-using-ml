import pickle
import numpy as np
import pandas as pd

def load_model_and_scaler():
    """Load the trained model and scaler"""
    model = pickle.load(open('crop_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Make crop prediction based on input parameters using a Random Forest model
    (Accuracy: 90%, Cross-validation: 88.5%)

    Feature Importance:
    ------------------
    - Rainfall: 18.36%
    - Humidity: 17.89%
    - Nitrogen: 16.45%
    - pH: 15.51%
    - Potassium: 14.11%
    - Temperature: 9.73%
    - Phosphorus: 7.95%

    Parameters:
    -----------
    N : float/int
        Nitrogen content in soil (0-140)
    P : float/int
        Phosphorus content in soil (5-145)
    K : float/int
        Potassium content in soil (5-205)
    temperature : float/int
        Temperature in Celsius (8-45)
    humidity : float/int
        Relative humidity percentage (14-100)
    ph : float/int
        Soil pH value (3.5-10)
    rainfall : float/int
        Rainfall in mm (20-300)

    Returns:
    --------
    tuple: (predicted_crop, confidence)
        predicted_crop: str - Name of the predicted crop
        confidence: float - Prediction confidence (0-1)
    """
    # Type validation
    params = {'N': N, 'P': P, 'K': K, 'temperature': temperature, 
             'humidity': humidity, 'ph': ph, 'rainfall': rainfall}
    
    for param_name, value in params.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"{param_name} must be a number, got {type(value).__name__}")
    
    # Range validation
    validations = {
        'N': (0, 140),
        'P': (5, 145),
        'K': (5, 205),
        'temperature': (8, 45),
        'humidity': (14, 100),
        'ph': (3.5, 10),
        'rainfall': (20, 300)
    }
    
    for param_name, (min_val, max_val) in validations.items():
        value = params[param_name]
        if not (min_val <= value <= max_val):
            raise ValueError(
                f"{param_name} should be between {min_val} and {max_val}, got {value}"
            )
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Create DataFrame with proper feature names
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                             columns=features)
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)
    
    # Get the highest probability
    max_prob = np.max(probabilities)
    
    return prediction[0], max_prob

# Test the model with sample data
if __name__ == "__main__":
    # Example input values
    sample_input = {
        'N': 90,
        'P': 42,
        'K': 43,
        'temperature': 20.87,
        'humidity': 82.00,
        'ph': 6.5,
        'rainfall': 202.935
    }
    
    # Make prediction
    predicted_crop, confidence = predict_crop(
        sample_input['N'], sample_input['P'], sample_input['K'],
        sample_input['temperature'], sample_input['humidity'],
        sample_input['ph'], sample_input['rainfall']
    )
    
    print(f"\nInput Parameters:")
    for param, value in sample_input.items():
        print(f"{param}: {value}")
    
    print(f"\nPredicted Crop: {predicted_crop}")
    print(f"Confidence: {confidence:.2%}")