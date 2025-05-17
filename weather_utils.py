import requests
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta

def get_weather_data(location, api_key, units="metric"):
    """Fetch weather data for a given location"""
    try:
        geolocator = Nominatim(user_agent="crop_recommendation_app_v1")
        location_data = geolocator.geocode(location, timeout=15)
        
        if location_data:
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={location_data.latitude}&lon={location_data.longitude}&appid={api_key}&units={units}"
            response = requests.get(weather_url, timeout=10)
            response.raise_for_status()
            
            return {
                'success': True,
                'weather_data': response.json(),
                'location_data': location_data
            }
        else:
            return {
                'success': False,
                'error': 'Location not found'
            }
    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': 'request'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': 'general'
        }

def convert_temperature(temp, from_unit="F", to_unit="C"):
    """Convert temperature between Fahrenheit and Celsius
    
    Args:
        temp (float/int): Temperature value to convert
        from_unit (str): Source unit ('F' or 'C')
        to_unit (str): Target unit ('F' or 'C')
        
    Returns:
        float: Converted temperature value
        
    Raises:
        ValueError: If temperature is not a number or units are invalid
    """
    # Validate input type
    try:
        temp = float(temp)
    except (TypeError, ValueError):
        raise ValueError("Temperature must be a numeric value")
        
    # Validate units
    valid_units = ['F', 'C']
    if from_unit not in valid_units or to_unit not in valid_units:
        raise ValueError("Units must be either 'F' or 'C'")
    
    # Perform conversion
    if from_unit == "F" and to_unit == "C":
        return round((temp - 32) * 5/9, 2)
    elif from_unit == "C" and to_unit == "F":
        return round((temp * 9/5) + 32, 2)
    return temp

def get_weather_alerts(weather_data):
    """Generate weather alerts based on current conditions
    
    Args:
        weather_data (dict): Weather data from OpenWeatherMap API
        
    Returns:
        list: List of weather alerts and farming recommendations
    """
    alerts = []
    if not weather_data or 'main' not in weather_data:
        return alerts
        
    temp = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    
    # Temperature-based alerts
    if temp < 0:
        alerts.append({
            'type': 'danger',
            'message': '❄️ Frost Alert: Protect sensitive crops!',
            'recommendation': 'Use frost protection methods like row covers or sprinklers'
        })
    elif temp > 35:
        alerts.append({
            'type': 'warning',
            'message': '🔥 Heat Stress Alert: Ensure adequate irrigation!',
            'recommendation': 'Increase watering frequency and consider shade protection'
        })
        
    # Humidity-based alerts
    if humidity < 30:
        alerts.append({
            'type': 'warning',
            'message': '⚠️ Low Humidity Alert: Consider additional irrigation!',
            'recommendation': 'Use mulching to retain soil moisture'
        })
    elif humidity > 80:
        alerts.append({
            'type': 'warning',
            'message': '💧 High Humidity Alert: Monitor for fungal diseases!',
            'recommendation': 'Ensure good air circulation and consider fungicide application'
        })
        
    return alerts

def get_farming_recommendations(weather_data):
    """Generate farming recommendations based on weather conditions
    
    Args:
        weather_data (dict): Weather data from OpenWeatherMap API
        
    Returns:
        dict: Farming recommendations for different activities
    """
    if not weather_data or 'main' not in weather_data:
        return {}
        
    temp = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    wind_speed = weather_data.get('wind', {}).get('speed', 0)
    
    recommendations = {
        'irrigation': {
            'suitable': 15 <= temp <= 30 and humidity < 70 and wind_speed < 20,
            'message': 'Favorable conditions for irrigation' if 15 <= temp <= 30 else 'Consider adjusting irrigation timing'
        },
        'fertilization': {
            'suitable': 10 <= temp <= 25 and 40 <= humidity <= 70,
            'message': 'Good conditions for fertilizer application' if 10 <= temp <= 25 else 'Delay fertilizer application'
        },
        'pest_control': {
            'suitable': 15 <= temp <= 25 and wind_speed < 10,
            'message': 'Suitable conditions for pest control' if wind_speed < 10 else 'Wind speed too high for spraying'
        }
    }
    
    return recommendations

def get_weekly_weather_summary(forecast_data):
    """Generate a weekly weather summary from forecast data
    
    Args:
        forecast_data (dict): Forecast data from OpenWeatherMap API
        
    Returns:
        dict: Weekly weather summary with farming implications
    """
    if not forecast_data or 'list' not in forecast_data:
        return {}
        
    daily_summaries = {}
    for item in forecast_data['list']:
        date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
        if date not in daily_summaries:
            daily_summaries[date] = {
                'temp_min': item['main']['temp_min'],
                'temp_max': item['main']['temp_max'],
                'humidity': item['main']['humidity'],
                'description': item['weather'][0]['description'],
                'farming_suitable': 15 <= item['main']['temp'] <= 30 and item['main']['humidity'] < 80
            }
    
    return daily_summaries