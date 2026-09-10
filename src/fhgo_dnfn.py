import torch
import torch.nn as nn
from src.dnf_network import DeepNeuroFuzzyNetwork
from src.fhgo import HenryGasOptimization

class FHGODNFNWrapper(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.dnfn = DeepNeuroFuzzyNetwork(input_dim)
        self.hgso = HenryGasOptimization(self.dnfn)

    def forward(self, x):
        return self.dnfn(x)

    def optimize(self, epoch):
        self.hgso.step(epoch)