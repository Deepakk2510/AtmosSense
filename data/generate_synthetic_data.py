import pandas as pd
import numpy as np
import os

def generate_data(num_samples=5000):
    np.random.seed(42)
    
    # Generate random features
    temperature = np.random.normal(loc=25, scale=10, size=num_samples) # Celsius
    humidity = np.random.normal(loc=60, scale=20, size=num_samples) # Percentage
    humidity = np.clip(humidity, 0, 100)
    wind_speed = np.random.normal(loc=15, scale=8, size=num_samples) # km/h
    wind_speed = np.clip(wind_speed, 0, 150)
    pressure = np.random.normal(loc=1013, scale=10, size=num_samples) # hPa
    
    # Target variable: Will it rain? 
    # High humidity, low pressure, and moderate temps usually mean rain.
    rain_prob = (humidity / 100) * 0.5 + ((1020 - pressure) / 40) * 0.3 + (np.clip(temperature, 0, 30) / 30) * 0.2
    # Add some noise
    rain_prob += np.random.normal(0, 0.1, size=num_samples)
    
    will_rain = (rain_prob > 0.55).astype(int)
    
    # Target variable for regression: Next day's temperature
    # Highly correlated with today's temp + some noise
    next_day_temp = temperature + np.random.normal(0, 3, size=num_samples)
    
    df = pd.DataFrame({
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure,
        'will_rain': will_rain,
        'next_day_temp': next_day_temp
    })
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    csv_path = os.path.join('data', 'historical_weather.csv')
    df.to_csv(csv_path, index=False)
    print(f"Generated {num_samples} samples and saved to {csv_path}")
    print("\nSample Data:")
    print(df.head())
    print(f"\nRain distribution:\n{df['will_rain'].value_counts(normalize=True)}")

if __name__ == "__main__":
    generate_data()
