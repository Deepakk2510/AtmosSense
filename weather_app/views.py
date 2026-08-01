import os
import requests
import joblib
import numpy as np
import torch
from django.shortcuts import render
from .ml_models_def import WeatherAutoencoder, RainTransferModel, TempTransferModel

def index(request):
    weather_data = None
    prediction_data = None
    error_message = None

    if request.method == 'POST':
        city = request.POST.get('city')
        api_key = os.getenv('OPENWEATHERMAP_API_KEY')
        
        if not api_key:
            error_message = "API Key is missing. Check your .env file."
        elif city:
            # Fetch data from OpenWeatherMap
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract features
                temp = data['main']['temp']
                humidity = data['main']['humidity']
                wind_speed = data['wind']['speed'] * 3.6 # m/s to km/h
                pressure = data['main']['pressure']
                weather_desc = data['weather'][0]['description'].capitalize()
                weather_icon = data['weather'][0]['icon']
                
                weather_data = {
                    'city': data['name'],
                    'temp': round(temp, 1),
                    'humidity': humidity,
                    'wind_speed': round(wind_speed, 1),
                    'pressure': pressure,
                    'description': weather_desc,
                    'icon': f"http://openweathermap.org/img/wn/{weather_icon}@2x.png"
                }
                
                # Load models and predict using Transfer Learning Models
                try:
                    # 1. Load Scaler and Scale Inputs
                    scaler = joblib.load('ml_models/scaler.pkl')
                    features = np.array([[temp, humidity, wind_speed, pressure]])
                    features_scaled = scaler.transform(features).astype(np.float32)
                    input_tensor = torch.tensor(features_scaled)

                    # 2. Instantiate Base Encoder
                    base_autoencoder = WeatherAutoencoder()

                    # 3. Load and Predict: Rain Classification
                    rain_model = RainTransferModel(base_autoencoder.encoder)
                    rain_model.load_state_dict(torch.load('ml_models/rain_transfer_model.pth', weights_only=True))
                    rain_model.eval()
                    with torch.no_grad():
                        rain_prob = rain_model(input_tensor).item()
                        will_rain = rain_prob > 0.5
                        
                    # 4. Load and Predict: Temperature Regression
                    temp_model = TempTransferModel(base_autoencoder.encoder)
                    temp_model.load_state_dict(torch.load('ml_models/temp_transfer_model.pth', weights_only=True))
                    temp_model.eval()
                    with torch.no_grad():
                        next_temp = temp_model(input_tensor).item()
                    
                    # Generate some dummy hourly predictions for the chart based on the regressor output
                    hourly_labels = ['Now', '+1h', '+2h', '+3h', '+4h', '+5h']
                    trend = (next_temp - temp) / 5
                    hourly_temps = [round(temp + (trend * i), 1) for i in range(6)]
                    
                    prediction_data = {
                        'will_rain': will_rain,
                        'next_day_temp': round(next_temp, 1),
                        'hourly_labels': hourly_labels,
                        'hourly_temps': hourly_temps
                    }
                except Exception as e:
                    error_message = f"Error loading PyTorch Transfer Learning models: {e}"
            else:
                error_message = f"City not found or API error (Status code: {response.status_code})."
                
    context = {
        'weather_data': weather_data,
        'prediction_data': prediction_data,
        'error_message': error_message
    }
    
    return render(request, 'weather_app/index.html', context)
