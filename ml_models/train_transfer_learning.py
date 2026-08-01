import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Step 1: Define the Base Model (Autoencoder) ---
class WeatherAutoencoder(nn.Module):
    def __init__(self):
        super(WeatherAutoencoder, self).__init__()
        # Encoder (This acts as our "Base" feature extractor)
        self.encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        # Decoder (Only used during pre-training)
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- Step 2: Define Transfer Learning Models ---
class RainTransferModel(nn.Module):
    def __init__(self, base_encoder):
        super(RainTransferModel, self).__init__()
        self.base_encoder = base_encoder
        self.classifier_head = nn.Sequential(
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.base_encoder(x)
        return self.classifier_head(x)

class TempTransferModel(nn.Module):
    def __init__(self, base_encoder):
        super(TempTransferModel, self).__init__()
        self.base_encoder = base_encoder
        self.regressor_head = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.base_encoder(x)
        return self.regressor_head(x)

def main():
    print("Loading synthetic historical data...")
    df = pd.read_csv(os.path.join('data', 'historical_weather.csv'))
    
    X = df[['temperature', 'humidity', 'wind_speed', 'pressure']].values
    y_class = df['will_rain'].values.astype(np.float32).reshape(-1, 1)
    y_reg = df['next_day_temp'].values.astype(np.float32).reshape(-1, 1)
    
    # Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    
    # Split
    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X_scaled, y_class, y_reg, test_size=0.2, random_state=42
    )
    
    # PyTorch DataLoaders
    train_loader_base = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(X_train)), batch_size=32, shuffle=True)
    train_loader_class = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(yc_train)), batch_size=32, shuffle=True)
    train_loader_reg = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(yr_train)), batch_size=32, shuffle=True)
    
    print("\n--- Phase 1: Pre-training Base Model (Autoencoder) ---")
    base_model = WeatherAutoencoder()
    criterion_base = nn.MSELoss()
    optimizer_base = optim.Adam(base_model.parameters(), lr=0.01)
    
    for epoch in range(10):
        for inputs, targets in train_loader_base:
            optimizer_base.zero_grad()
            outputs = base_model(inputs)
            loss = criterion_base(outputs, targets)
            loss.backward()
            optimizer_base.step()
    print("Base Model pre-trained. Learned fundamental weather representations.")
    
    print("\n--- Phase 2: Transfer Learning (Fine-Tuning) ---")
    # Freeze the base encoder weights to demonstrate true transfer learning
    for param in base_model.encoder.parameters():
        param.requires_grad = False
        
    print("Training Rain Classification Head...")
    rain_model = RainTransferModel(base_model.encoder)
    criterion_class = nn.BCELoss()
    # Only train the new head!
    optimizer_class = optim.Adam(rain_model.classifier_head.parameters(), lr=0.01)
    
    for epoch in range(15):
        for inputs, targets in train_loader_class:
            optimizer_class.zero_grad()
            outputs = rain_model(inputs)
            loss = criterion_class(outputs, targets)
            loss.backward()
            optimizer_class.step()
            
    # Eval Class
    rain_model.eval()
    with torch.no_grad():
        preds = rain_model(torch.tensor(X_test))
        acc = ((preds > 0.5) == torch.tensor(yc_test)).float().mean().item()
        print(f"Transfer Classifier Accuracy: {acc*100:.2f}%")
        
    print("\nTraining Temperature Regression Head...")
    temp_model = TempTransferModel(base_model.encoder)
    criterion_reg = nn.MSELoss()
    optimizer_reg = optim.Adam(temp_model.regressor_head.parameters(), lr=0.01)
    
    for epoch in range(15):
        for inputs, targets in train_loader_reg:
            optimizer_reg.zero_grad()
            outputs = temp_model(inputs)
            loss = criterion_reg(outputs, targets)
            loss.backward()
            optimizer_reg.step()
            
    # Eval Reg
    temp_model.eval()
    with torch.no_grad():
        preds = temp_model(torch.tensor(X_test))
        mae = torch.abs(preds - torch.tensor(yr_test)).mean().item()
        print(f"Transfer Regressor MAE: {mae:.2f} degrees")
        
    print("\n--- Saving PyTorch Transfer Learning Models ---")
    os.makedirs('ml_models', exist_ok=True)
    
    # Save state dicts
    torch.save(rain_model.state_dict(), os.path.join('ml_models', 'rain_transfer_model.pth'))
    torch.save(temp_model.state_dict(), os.path.join('ml_models', 'temp_transfer_model.pth'))
    joblib.dump(scaler, os.path.join('ml_models', 'scaler.pkl'))
    
    print("PyTorch Models and scaler saved successfully!")

if __name__ == '__main__':
    main()
