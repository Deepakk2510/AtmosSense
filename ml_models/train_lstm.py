import pandas as pd
import numpy as np
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class WeatherLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=2, output_size=1, is_classification=False):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.is_classification = is_classification
        if is_classification:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        out, _ = self.lstm(x)
        # We only care about the output of the last time step
        out = out[:, -1, :] 
        out = self.fc(out)
        if self.is_classification:
            out = self.sigmoid(out)
        return out

def create_sequences(data, targets, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(targets[i + seq_length])
    return np.array(xs), np.array(ys)

def main():
    print("Loading real-world historical data...")
    df = pd.read_csv(os.path.join('data', 'real_weather_data.csv'))
    
    # 4 features
    features = df[['temperature', 'precipitation', 'wind_speed', 'radiation']].values
    
    # Targets
    y_class = df['will_rain'].values.astype(np.float32).reshape(-1, 1)
    y_reg = df['next_day_temp'].values.astype(np.float32).reshape(-1, 1)
    
    # Scale Data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features).astype(np.float32)
    
    seq_length = 7
    X_seq, yc_seq = create_sequences(features_scaled, y_class, seq_length)
    _, yr_seq = create_sequences(features_scaled, y_reg, seq_length)
    
    # Split into train and test
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    yc_train, yc_test = yc_seq[:split], yc_seq[split:]
    yr_train, yr_test = yr_seq[:split], yr_seq[split:]
    
    print(f"Training sequences: {len(X_train)}")
    
    # Loaders
    batch_size = 64
    train_loader_class = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(yc_train)), batch_size=batch_size, shuffle=True)
    train_loader_reg = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(yr_train)), batch_size=batch_size, shuffle=True)
    
    # --- Train Rain LSTM (Classification) ---
    print("\nTraining Rain LSTM...")
    rain_model = WeatherLSTM(input_size=4, hidden_size=32, num_layers=2, output_size=1, is_classification=True)
    criterion_class = nn.BCELoss()
    optimizer_class = optim.Adam(rain_model.parameters(), lr=0.005)
    
    for epoch in range(15):
        for inputs, targets in train_loader_class:
            optimizer_class.zero_grad()
            outputs = rain_model(inputs)
            loss = criterion_class(outputs, targets)
            loss.backward()
            optimizer_class.step()
            
    # Eval
    rain_model.eval()
    with torch.no_grad():
        preds = rain_model(torch.tensor(X_test))
        acc = ((preds > 0.5) == torch.tensor(yc_test)).float().mean().item()
        print(f"LSTM Rain Classifier Accuracy: {acc*100:.2f}%")
        
    # --- Train Temperature LSTM (Regression) ---
    print("\nTraining Temperature LSTM...")
    temp_model = WeatherLSTM(input_size=4, hidden_size=32, num_layers=2, output_size=1, is_classification=False)
    criterion_reg = nn.MSELoss()
    optimizer_reg = optim.Adam(temp_model.parameters(), lr=0.005)
    
    for epoch in range(20):
        for inputs, targets in train_loader_reg:
            optimizer_reg.zero_grad()
            outputs = temp_model(inputs)
            loss = criterion_reg(outputs, targets)
            loss.backward()
            optimizer_reg.step()
            
    # Eval
    temp_model.eval()
    with torch.no_grad():
        preds = temp_model(torch.tensor(X_test))
        mae = torch.abs(preds - torch.tensor(yr_test)).mean().item()
        print(f"LSTM Temperature Regressor MAE: {mae:.2f} degrees")
        
    print("\n--- Saving LSTM Models ---")
    os.makedirs('ml_models', exist_ok=True)
    torch.save(rain_model.state_dict(), os.path.join('ml_models', 'lstm_rain.pth'))
    torch.save(temp_model.state_dict(), os.path.join('ml_models', 'lstm_temp.pth'))
    joblib.dump(scaler, os.path.join('ml_models', 'lstm_scaler.pkl'))
    
    print("Models saved successfully!")

if __name__ == '__main__':
    main()
