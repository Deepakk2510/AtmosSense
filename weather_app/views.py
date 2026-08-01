import os
import requests
import joblib
import numpy as np
import torch
from django.shortcuts import render
from .ml_models_def import WeatherLSTM

def index(request):
    weather_data = None
    prediction_data = None
    error_message = None

    if request.method == 'POST':
        city = request.POST.get('city')
        
        if city:
            try:
                # 1. Geocoding: Get Lat/Lon for the city
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
                geo_resp = requests.get(geo_url).json()
                
                if 'results' not in geo_resp:
                    error_message = f"City '{city}' not found."
                else:
                    lat = geo_resp['results'][0]['latitude']
                    lon = geo_resp['results'][0]['longitude']
                    city_name = geo_resp['results'][0]['name']
                    
                    # 2. Fetch last 7 days of weather + current day
                    weather_url = (
                        f"https://api.open-meteo.com/v1/forecast?"
                        f"latitude={lat}&longitude={lon}&past_days=7&forecast_days=1&"
                        f"daily=temperature_2m_max,precipitation_sum,windspeed_10m_max,shortwave_radiation_sum&"
                        f"timezone=auto"
                    )
                    weather_resp = requests.get(weather_url).json()
                    
                    daily = weather_resp['daily']
                    
                    # Today's actual weather for the UI
                    today_idx = -1
                    today_temp = daily['temperature_2m_max'][today_idx]
                    today_rain = daily['precipitation_sum'][today_idx]
                    today_wind = daily['windspeed_10m_max'][today_idx]
                    today_rad = daily['shortwave_radiation_sum'][today_idx]
                    
                    weather_data = {
                        'city': city_name,
                        'temp': round(today_temp, 1),
                        'humidity': "N/A (Open-Meteo Daily)", # Daily humidity isn't directly exposed
                        'wind_speed': round(today_wind, 1),
                        'pressure': "N/A",
                        'description': "Rainy" if today_rain > 1.0 else "Clear/Cloudy",
                        'icon': "http://openweathermap.org/img/wn/10d@2x.png" if today_rain > 1.0 else "http://openweathermap.org/img/wn/01d@2x.png"
                    }
                    
                    # 3. Prepare LSTM Sequence (Past 7 days)
                    # We need exactly 7 days. The API returned 8 days (7 past + 1 forecast).
                    # We take the first 7 days (index 0 to 6)
                    seq_temp = daily['temperature_2m_max'][:7]
                    seq_rain = daily['precipitation_sum'][:7]
                    seq_wind = daily['windspeed_10m_max'][:7]
                    seq_rad = daily['shortwave_radiation_sum'][:7]
                    
                    features = []
                    for i in range(7):
                        features.append([seq_temp[i], seq_rain[i], seq_wind[i], seq_rad[i]])
                    features = np.array(features) # Shape: (7, 4)
                    
                    # 4. Load Scaler and Scale
                    scaler = joblib.load('ml_models/lstm_scaler.pkl')
                    features_scaled = scaler.transform(features).astype(np.float32)
                    input_tensor = torch.tensor(features_scaled).unsqueeze(0) # Shape: (1, 7, 4)
                    
                    # 5. Load and Predict: Rain Classification LSTM
                    rain_model = WeatherLSTM(input_size=4, hidden_size=32, num_layers=2, output_size=1, is_classification=True)
                    rain_model.load_state_dict(torch.load('ml_models/lstm_rain.pth', weights_only=True))
                    rain_model.eval()
                    with torch.no_grad():
                        rain_prob = rain_model(input_tensor).item()
                        will_rain = rain_prob > 0.5
                        
                    # 6. Load and Predict: Temperature Regression LSTM
                    temp_model = WeatherLSTM(input_size=4, hidden_size=32, num_layers=2, output_size=1, is_classification=False)
                    temp_model.load_state_dict(torch.load('ml_models/lstm_temp.pth', weights_only=True))
                    temp_model.eval()
                    with torch.no_grad():
                        next_temp = temp_model(input_tensor).item()
                        
                    hourly_labels = ['Now', '+1h', '+2h', '+3h', '+4h', '+5h']
                    trend = (next_temp - today_temp) / 5
                    hourly_temps = [round(today_temp + (trend * i), 1) for i in range(6)]
                    
                    prediction_data = {
                        'will_rain': will_rain,
                        'next_day_temp': round(next_temp, 1),
                        'hourly_labels': hourly_labels,
                        'hourly_temps': hourly_temps
                    }
                    
            except Exception as e:
                error_message = f"Error generating LSTM forecast: {str(e)}"
                
    context = {
        'weather_data': weather_data,
        'prediction_data': prediction_data,
        'error_message': error_message
    }
    
    return render(request, 'weather_app/index.html', context)
