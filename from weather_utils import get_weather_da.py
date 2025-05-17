from weather_utils import get_weather_data
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENWEATHERMAP_API_KEY")

# Use the weather utility function
weather_result = get_weather_data("noida", api_key)