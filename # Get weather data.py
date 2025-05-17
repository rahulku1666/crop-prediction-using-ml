# Get weather data
weather_result = get_weather_data("noida", api_key)

if weather_result['success']:
    # Get weather alerts
    alerts = get_weather_alerts(weather_result['weather_data'])
    
    # Get farming recommendations
    recommendations = get_farming_recommendations(weather_result['weather_data'])
    
    # Get weekly summary (if you have forecast data)
    weekly_summary = get_weekly_weather_summary(forecast_data)