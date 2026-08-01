import requests
import pandas as pd
import os

def fetch_historical_weather():
    print("Fetching 10 years of historical weather data for London (Open-Meteo)...")
    
    # London coordinates
    lat = 51.5085
    lon = -0.1257
    
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&"
        f"start_date=2014-01-01&end_date=2024-01-01&"
        f"daily=temperature_2m_mean,precipitation_sum,windspeed_10m_max,shortwave_radiation_sum&"
        f"timezone=Europe/London"
    )
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        daily = data['daily']
        
        df = pd.DataFrame({
            'date': daily['time'],
            'temperature': daily['temperature_2m_mean'],
            'precipitation': daily['precipitation_sum'],
            'wind_speed': daily['windspeed_10m_max'],
            'radiation': daily['shortwave_radiation_sum']
        })
        
        # Clean nulls (if any)
        df = df.dropna()
        
        # Create Target Variables (Shifted by 1 day)
        # Next day's temperature
        df['next_day_temp'] = df['temperature'].shift(-1)
        # Next day will rain if precipitation > 1.0mm
        df['will_rain'] = (df['precipitation'].shift(-1) > 1.0).astype(int)
        
        # Drop the last row since it won't have a 'next_day' target
        df = df.dropna()
        
        os.makedirs('data', exist_ok=True)
        csv_path = os.path.join('data', 'real_weather_data.csv')
        df.to_csv(csv_path, index=False)
        print(f"Successfully saved {len(df)} days of real weather data to {csv_path}!")
    else:
        print(f"Failed to fetch data: {response.status_code}")
        print(response.text)

if __name__ == '__main__':
    fetch_historical_weather()
