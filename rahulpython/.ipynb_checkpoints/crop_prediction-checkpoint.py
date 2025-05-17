import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib  # Added missing import

# Load and prepare the dataset
def load_data(filepath):
    data = pd.read_csv(filepath)
    X = data.drop('crop', axis=1)  # Assuming 'crop' is the target column
    y = data['crop']
    return X, y

# Create base models
def create_base_models():
    models = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'svm': SVC(kernel='rbf', probability=True, random_state=42)
    }
    return models

# Create stacking model
class CropPredictor:
    def __init__(self):
        self.base_models = create_base_models()
        self.meta_model = LogisticRegression(multi_class='ovr')  # Modified for multiclass
        self.scaler = StandardScaler()
        self.feature_importance = {}
    
    def fit(self, X, y):
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train base models and get predictions
        meta_features = np.zeros((X_val.shape[0], len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            model.fit(X_train, y_train)
            # Modified to handle multiclass predictions
            proba = model.predict_proba(X_val)
            meta_features[:, i] = np.mean(proba, axis=1)
        
        # Train meta model
        self.meta_model.fit(meta_features, y_val)
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        meta_features = np.zeros((X_scaled.shape[0], len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            # Modified to handle multiclass predictions
            proba = model.predict_proba(X_scaled)
            meta_features[:, i] = np.mean(proba, axis=1)
        
        return self.meta_model.predict(meta_features)

    def get_feature_importance(self, feature_names):
        """Calculate feature importance from Random Forest model"""
        rf_model = self.base_models['rf']
        importances = rf_model.feature_importances_
        self.feature_importance = dict(zip(feature_names, importances))
        return self.feature_importance

    def save_model(self, filepath):
        """Save the trained model to disk"""
        joblib.dump(self, filepath)

    @staticmethod
    def load_model(filepath):
        """Load a trained model from disk"""
        return joblib.load(filepath)

    def tune_hyperparameters(self, X, y):
        """Tune hyperparameters for base models"""
        param_grid = {
            'rf': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20]
            }
        }
        
        for name, model in self.base_models.items():
            if name in param_grid:
                grid_search = GridSearchCV(model, param_grid[name], cv=5)
                grid_search.fit(X, y)
                self.base_models[name] = grid_search.best_estimator_

# Main execution
if __name__ == "__main__":
    try:
        # Load and prepare data
        X, y = load_data('crop_data.csv')
        
        if X is None or y is None:
            print("Error: Failed to load data")
            exit(1)
            
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train the stacking model
        crop_predictor = CropPredictor()
        
        # Tune hyperparameters (optional)
        crop_predictor.tune_hyperparameters(X_train, y_train)
        
        # Train the model
        crop_predictor.fit(X_train, y_train)
        
        # Get feature importance
        feature_importance = crop_predictor.get_feature_importance(X.columns)
        print("\nFeature Importance:")
        for feature, importance in sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")
        
        # Perform cross-validation
        cv_scores = cross_val_score(crop_predictor.base_models['rf'], 
                                  X_train, y_train, cv=5)
        print(f"\nCross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # Make predictions
        predictions = crop_predictor.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nModel Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
    except Exception as e:
        print(f"An error occurred: {str(e)}")
