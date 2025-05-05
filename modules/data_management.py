import json
import pandas as pd
from datetime import datetime

def save_prediction(prediction_data, filename='prediction_history.json'):
    try:
        try:
            with open(filename, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []
        
        history.append(prediction_data)
        
        with open(filename, 'w') as f:
            json.dump(history, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False

def load_predictions(filename='prediction_history.json'):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []