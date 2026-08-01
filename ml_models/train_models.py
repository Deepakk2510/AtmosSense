import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import joblib
import os

def train_models():
    print("Loading synthetic historical data...")
    # Load dataset
    data_path = os.path.join('data', 'historical_weather.csv')
    df = pd.read_csv(data_path)
    
    # Features (X) and Targets (y)
    X = df[['temperature', 'humidity', 'wind_speed', 'pressure']]
    y_class = df['will_rain'] # For Classification
    y_reg = df['next_day_temp'] # For Regression
    
    print("\n--- Training Random Forest Classifier (Rain Prediction) ---")
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_c, y_train_c)
    
    y_pred_c = clf.predict(X_test_c)
    acc = accuracy_score(y_test_c, y_pred_c)
    print(f"Classifier Accuracy: {acc * 100:.2f}%")
    
    print("\n--- Training Random Forest Regressor (Temperature Prediction) ---")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train_r, y_train_r)
    
    y_pred_r = reg.predict(X_test_r)
    mae = mean_absolute_error(y_test_r, y_pred_r)
    r2 = r2_score(y_test_r, y_pred_r)
    print(f"Regressor MAE (Mean Absolute Error): {mae:.2f} degrees")
    print(f"Regressor R2 Score: {r2:.4f}")
    
    print("\n--- Saving Models ---")
    os.makedirs('ml_models', exist_ok=True)
    clf_path = os.path.join('ml_models', 'rain_classifier.pkl')
    reg_path = os.path.join('ml_models', 'temp_regressor.pkl')
    
    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)
    
    print(f"Saved Rain Classifier to: {clf_path}")
    print(f"Saved Temp Regressor to: {reg_path}")
    print("Done Phase 2!")

if __name__ == '__main__':
    train_models()
