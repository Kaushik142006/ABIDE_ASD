import math
import torch


class HenryGasOptimization:
    def __init__(self, model, max_epochs=25):
        self.model = model
        self.max_epochs = max_epochs

        device = model.centers.device

        self.solubility = torch.ones_like(
            model.centers,
            device=device
        )

        self.partial_pressure = torch.rand_like(
            model.centers,
            device=device
        )

    def step(self, epoch):
        with torch.no_grad():

            # Always keep optimizer tensors on the model's device.
            device = self.model.centers.device

            self.solubility = self.solubility.to(device)
            self.partial_pressure = self.partial_pressure.to(device)

            T = math.exp(-epoch / self.max_epochs)

            self.solubility *= math.exp(
                -1.0 / (T + 1e-6)
            )

            diffusion = (
                torch.randn_like(self.model.centers)
                * self.solubility
                * self.partial_pressure
            )

            self.model.centers.add_(
                diffusion * 0.01
            )