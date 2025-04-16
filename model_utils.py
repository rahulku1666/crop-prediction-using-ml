def get_model_info():
    return {
        'feature_importance': {
            'Rainfall': {'value': 18.36, 'ci_lower': 17.2, 'ci_upper': 19.5},
            'Humidity': {'value': 17.89, 'ci_lower': 16.8, 'ci_upper': 19.0},
            'Nitrogen': {'value': 16.45, 'ci_lower': 15.3, 'ci_upper': 17.6},
            'pH': {'value': 15.51, 'ci_lower': 14.4, 'ci_upper': 16.6},
            'Potassium': {'value': 14.11, 'ci_lower': 13.0, 'ci_upper': 15.2},
            'Temperature': {'value': 9.73, 'ci_lower': 8.6, 'ci_upper': 10.9},
            'Phosphorus': {'value': 7.95, 'ci_lower': 6.8, 'ci_upper': 9.1}
        },
        'performance': {
            'accuracy': 0.90,
            'cv_score': 0.885,
            'precision': 0.88,
            'recall': 0.87,
            'f1_score': 0.89,
            'model_type': 'Random Forest Classifier',
            'last_updated': '2024-01-20',
            'training_samples': 10000,
            'validation_method': '5-fold cross validation'
        },
        'recommendations': {
            'Rainfall': {'low': 50, 'optimal': 200, 'high': 300},
            'Humidity': {'low': 30, 'optimal': 70, 'high': 90},
            'Temperature': {'low': 10, 'optimal': 25, 'high': 35},
            'pH': {'low': 5.5, 'optimal': 6.5, 'high': 7.5},
            'Nitrogen': {'low': 40, 'optimal': 90, 'high': 120},
            'Phosphorus': {'low': 20, 'optimal': 42, 'high': 80},
            'Potassium': {'low': 20, 'optimal': 43, 'high': 100}
        },
        'model_version': '2.0',
        'update_frequency': 'Monthly'
    }

def analyze_parameters(current_values):
    """Analyze current parameter values against recommendations"""
    model_info = get_model_info()
    analysis = {}
    
    for param, value in current_values.items():
        if param in model_info['recommendations']:
            rec = model_info['recommendations'][param]
            if value < rec['low']:
                analysis[param] = {'status': 'low', 'optimal': rec['optimal']}
            elif value > rec['high']:
                analysis[param] = {'status': 'high', 'optimal': rec['optimal']}
            else:
                analysis[param] = {'status': 'optimal', 'optimal': rec['optimal']}
    
    return analysis

def get_parameter_importance():
    """Get sorted parameter importance"""
    model_info = get_model_info()
    importance = [(k, v['value']) for k, v in model_info['feature_importance'].items()]
    return sorted(importance, key=lambda x: x[1], reverse=True)

def get_model_performance():
    """Get model performance metrics"""
    return get_model_info()['performance']

def get_fertilizer_recommendations(N, P, K):
    """Get fertilizer recommendations based on NPK values"""
    recommendations = []
    
    if N < 50:
        recommendations.append({
            'nutrient': 'Nitrogen',
            'fertilizer': 'Urea (46-0-0)',
            'message': '💡 Consider using Urea (46-0-0) to boost Nitrogen.'
        })
    
    if P < 20:
        recommendations.append({
            'nutrient': 'Phosphorus',
            'fertilizer': 'Di-Ammonium Phosphate (DAP)',
            'message': '💡 Use Di-Ammonium Phosphate (DAP) to improve Phosphorus.'
        })
    
    if K < 30:
        recommendations.append({
            'nutrient': 'Potassium',
            'fertilizer': 'Muriate of Potash (MOP)',
            'message': '💡 Apply Muriate of Potash (MOP) for Potassium enrichment.'
        })
    
    return recommendations

def save_prediction(prediction_data):
    """Save prediction data to history"""
    history_file = "prediction_history.json"
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(prediction_data)
        
        with open(history_file, 'w') as f:
            json.dump(history, f)
        return True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False