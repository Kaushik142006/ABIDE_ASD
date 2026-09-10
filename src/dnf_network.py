# 7
import torch
import torch.nn as nn

class DeepNeuroFuzzyNetwork(nn.Module):
    def __init__(self, input_dim, num_rules=32):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.centers = nn.Parameter(torch.randn(64, num_rules))
        self.sigmas = nn.Parameter(torch.ones(64, num_rules))
        self.classifier = nn.Sequential(
            nn.Linear(num_rules, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        dist = (feat.unsqueeze(-1) - self.centers.unsqueeze(0)) ** 2
        sigmas_sq = self.sigmas.unsqueeze(0) ** 2 + 1e-6
        mu = torch.exp(-dist / sigmas_sq)
        rule_firing = torch.sum(mu, dim=1)
        return self.classifier(rule_firing)