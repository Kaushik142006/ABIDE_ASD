import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 32))
        self.decoder = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, input_dim))

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return latent, recon

class FuzzyRecurrentNeuralNetwork(nn.Module):
    def __init__(self, num_nodes=50):
        super().__init__()
        self.sae = SparseAutoencoder(input_dim=num_nodes)
        self.fuzzy_centers = nn.Parameter(torch.zeros(32))
        self.fuzzy_widths = nn.Parameter(torch.ones(32))
        self.gru = nn.GRU(32, 32, num_layers=2, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x_2d):
        seq = x_2d.squeeze(1)
        latent, recon = self.sae(seq)
        membership = torch.exp(-((latent - self.fuzzy_centers) ** 2) / (self.fuzzy_widths ** 2 + 1e-6))
        fuzzified = latent * membership
        out, _ = self.gru(fuzzified)
        preds = self.classifier(out[:, -1, :])
        return preds, latent, recon