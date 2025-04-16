import requests
from geopy.geocoders import Nominatim

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
    """Convert temperature between Fahrenheit and Celsius"""
    if from_unit == "F" and to_unit == "C":
        return (temp - 32) * 5/9
    elif from_unit == "C" and to_unit == "F":
        return (temp * 9/5) + 32
    return temp