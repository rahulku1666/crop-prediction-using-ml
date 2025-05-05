import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import pickle

def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = pickle.load(open('crop_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        return model, scaler
    except FileNotFoundError:
        raise Exception("Model files not found. Please train the model first.")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def train_knn_model(data_path='Crop_recommendation.csv', n_neighbors=5):
    """Train KNN model for crop prediction"""
    try:
        # Load and prepare data
        print("Loading dataset...")
        df = pd.read_csv(data_path)
        print(f"Dataset shape: {df.shape}")
        print("Columns:", list(df.columns))
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train KNN model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        
        # Calculate metrics
        accuracy = knn.score(X_test, y_test)
        cv_scores = cross_val_score(knn, X_scaled, y, cv=5)
        
        # Save model and scaler
        pickle.dump(knn, open('crop_model.pkl', 'wb'))
        pickle.dump(scaler, open('scaler.pkl', 'wb'))
        
        return {
            'accuracy': accuracy,
            'cv_score': cv_scores.mean(),
            'model': knn,
            'scaler': scaler
        }
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """Make crop prediction using KNN model"""
    # Parameter validation
    params = {
        'N': (N, 0, 140, "Nitrogen"),
        'P': (P, 5, 145, "Phosphorus"),
        'K': (K, 5, 205, "Potassium"),
        'temperature': (temperature, 0, 45, "Temperature"),
        'humidity': (humidity, 0, 100, "Humidity"),
        'ph': (ph, 0, 14, "pH"),
        'rainfall': (rainfall, 0, 300, "Rainfall")
    }
    
    # Validate parameter ranges
    for param, (value, min_val, max_val, name) in params.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        if value < min_val or value > max_val:
            raise ValueError(f"{name} should be between {min_val} and {max_val}")
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    # Prepare input data
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], 
                             columns=features)
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)
    
    return prediction[0], np.max(probabilities)

# Test the model
if __name__ == "__main__":
    try:
        # Train the model
        print("Training KNN model...")
        results = train_knn_model()
        print(f"Model Accuracy: {results['accuracy']:.2%}")
        print(f"Cross-validation Score: {results['cv_score']:.2%}")
        
        # Define test input parameters
        N = 90  # Nitrogen content
        P = 42  # Phosphorus content
        K = 43  # Potassium content
        temperature = 20.87
        humidity = 82.00
        ph = 6.5
        rainfall = 202.935
        
        # Make prediction
        crop, confidence = predict_crop(N, P, K, temperature, humidity, ph, rainfall)
        print(f"\nPredicted Crop: {crop}")
        print(f"Confidence: {confidence:.2%}")
        
    except Exception as e:
        print(f"⚠️ Error: {str(e)}")