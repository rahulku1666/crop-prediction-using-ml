import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import joblib

def train_knn_model(X, y, n_neighbors=5):
    """Train KNN model for crop prediction"""
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    # Calculate performance metrics
    accuracy = knn.score(X_test, y_test)
    cv_scores = cross_val_score(knn, X_scaled, y, cv=5)
    
    # Save the model and scaler
    joblib.dump(knn, 'models/knn_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return {
        'model': knn,
        'scaler': scaler,
        'accuracy': accuracy,
        'cv_score': cv_scores.mean()
    }

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """Predict crop using trained KNN model"""
    try:
        # Load model and scaler
        knn = joblib.load('models/knn_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        # Prepare input data
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = knn.predict(input_scaled)
        confidence = max(knn.predict_proba(input_scaled)[0])
        
        return prediction[0], confidence
        
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")