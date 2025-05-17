import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib
import logging
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_data(data_path: str) -> tuple:
    """
    Load and prepare dataset for training.
    
    Args:
        data_path (str): Path to the CSV file containing crop data
        
    Returns:
        tuple: (X, y, label_encoder) containing features, labels and the encoder
        
    Raises:
        FileNotFoundError: If the data file is not found
        ValueError: If the data format is invalid
    """
    try:
        df = pd.read_csv(data_path)
        if 'label' not in df.columns:
            raise ValueError("Dataset must contain 'label' column")
            
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['label'])
        
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Save label encoder
        encoder_path = Path('label_encoder.pkl')
        joblib.dump(label_encoder, encoder_path)
        logger.info(f"Label encoder saved to {encoder_path}")
        
        return X, y, label_encoder
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def create_stacking_model() -> StackingClassifier:
    """
    Create and configure the stacking classifier model.
    
    Returns:
        StackingClassifier: Configured stacking model with base learners
    """
    try:
        base_learners = [
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('dt', DecisionTreeClassifier(random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
        ]
        
        meta_learner = LogisticRegression()
        
        return StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=5
        )
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

def save_model(model, filepath: str) -> None:
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model to save
        filepath (str): Path where to save the model
    """
    try:
        joblib.dump(model, filepath)
        logger.info(f"Model saved successfully to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(filepath: str):
    """
    Load a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model file
        
    Returns:
        The loaded model
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"Model loaded successfully from {filepath}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def analyze_feature_importance(model, X):
    """
    Analyze and visualize feature importance from the Random Forest base model.
    
    Args:
        model: Trained stacking model
        X: Feature DataFrame
    """
    try:
        # Extract feature importances from Random Forest
        importances = model.named_estimators_['rf'].feature_importances_
        features = X.columns
        
        # Sort importances
        indices = np.argsort(importances)[::-1]
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances in Crop Prediction", pad=20)
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), features[indices], rotation=45, ha='right')
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('feature_importance.png')
        plt.close()
        
        # Log feature importance information
        logger.info("\nFeature Importance Rankings:")
        for idx in indices:
            logger.info(f"{features[idx]}: {importances[idx]:.4f}")
            
    except Exception as e:
        logger.error(f"Error analyzing feature importance: {str(e)}")
        raise

def main():
    """
    Main function to train and evaluate the crop prediction model.
    """
    try:
        # Prepare data
        logger.info("Loading and preparing data...")
        X, y, label_encoder = prepare_data("crop_data.csv")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train model
        logger.info("Training model...")
        model = create_stacking_model()
        model.fit(X_train, y_train)
        
        # Save model with a more descriptive name
        model_path = 'stacking_crop_model.pkl'
        save_model(model, model_path)
        
        # Example of loading the model (commented out as it's for demonstration)
        # loaded_model = load_model('stacking_crop_model.pkl')
        
        # Evaluate model with detailed metrics and visualization
        logger.info("Evaluating model performance...")
        evaluate_model(model, X_test, y_test, label_encoder)
        
        # Analyze feature importance
        logger.info("Analyzing feature importance...")
        analyze_feature_importance(model, X)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate model performance with detailed metrics and visualizations.
    
    Args:
        model: Trained model to evaluate
        X_test: Test features
        y_test: True labels
        label_encoder: Fitted label encoder for class names
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        
        # Generate detailed classification report
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_test_labels = label_encoder.inverse_transform(y_test)
        report = classification_report(y_test_labels, y_pred_labels, 
                                    target_names=label_encoder.classes_)
        logger.info("\nClassification Report:\n" + report)
        
        # Generate and plot confusion matrix with improved styling
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', 
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_,
                   cmap='Blues')
        plt.title('Crop Prediction Confusion Matrix', pad=20)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info("Confusion matrix visualization saved as 'confusion_matrix.png'")
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise

def predict_crop(input_data: dict) -> str:
    """
    Predict crop type based on soil and environmental parameters.
    
    Args:
        input_data (dict): Dictionary containing soil and environmental parameters
            Required keys: 'N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall'
            
    Returns:
        str: Predicted crop name
        
    Raises:
        ValueError: If input data is missing required features
        Exception: For other prediction errors
    """
    try:
        # Validate input features
        required_features = ['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall']
        missing_features = [feat for feat in required_features if feat not in input_data]
        
        if missing_features:
            raise ValueError(f"Missing required features: {', '.join(missing_features)}")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure correct feature order
        input_df = input_df[required_features]
        
        # Make prediction
        pred_encoded = model.predict(input_df)
        pred_label = label_encoder.inverse_transform(pred_encoded)
        
        return pred_label[0]
        
    except ValueError as e:
        logger.error(f"Invalid input data: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise

# Example usage with error handling
try:
    new_sample = {
        'N': 90,
        'P': 42,
        'K': 43,
        'temperature': 20.87,
        'humidity': 82,
        'pH': 6.5,
        'rainfall': 202.93
    }
    predicted_crop = predict_crop(new_sample)
    logger.info(f"Predicted crop: {predicted_crop}")
    
except ValueError as e:
    logger.error(f"Input validation error: {str(e)}")
except Exception as e:
    logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()
