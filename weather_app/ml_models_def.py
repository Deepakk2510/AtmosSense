import torch
import torch.nn as nn

class WeatherLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=2, output_size=1, is_classification=False):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.is_classification = is_classification
        if is_classification:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        if self.is_classification:
            out = self.sigmoid(out)
        return out

# Old Transfer Learning Models below (kept for backwards compatibility if needed)
class WeatherAutoencoder(nn.Module):
    def __init__(self):
        super(WeatherAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

class RainTransferModel(nn.Module):
    def __init__(self, base_encoder):
        super(RainTransferModel, self).__init__()
        self.base_encoder = base_encoder
        self.classifier_head = nn.Sequential(
            nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.classifier_head(self.base_encoder(x))

class TempTransferModel(nn.Module):
    def __init__(self, base_encoder):
        super(TempTransferModel, self).__init__()
        self.base_encoder = base_encoder
        self.regressor_head = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.regressor_head(self.base_encoder(x))
