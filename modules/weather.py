import requests
from geopy.geocoders import Nominatim

def get_weather_data(location, api_key, units="metric"):
    try:
        geolocator = Nominatim(user_agent="crop_recommendation_app_v1")
        location_data = geolocator.geocode(location, timeout=15)
        
        if location_data:
            weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={location_data.latitude}&lon={location_data.longitude}&appid={api_key}&units={units}"
            response = requests.get(weather_url, timeout=10)
            response.raise_for_status()
            return response.json(), location_data.address
            
        return None, None
    except Exception as e:
        raise Exception(f"Weather data fetch error: {str(e)}")