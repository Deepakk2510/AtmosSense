import torch
import torch.nn as nn

class WeatherAutoencoder(nn.Module):
    def __init__(self):
        super(WeatherAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

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
