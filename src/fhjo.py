import torch

class JellyfishSearchOptimization:
    def __init__(self, mask_param, max_epochs=25):
        self.mask = mask_param
        self.max_epochs = max_epochs

    def step(self, epoch):
        with torch.no_grad():
            time_control = abs((1 - epoch / self.max_epochs) * (2 * torch.rand(1).item() - 1))
            if time_control > 0.5:
                ocean = (self.mask.mean() - self.mask) * torch.rand_like(self.mask)
                self.mask.add_(ocean * 0.05)
            else:
                swarm = torch.randn_like(self.mask) * 0.02
                self.mask.add_(swarm)