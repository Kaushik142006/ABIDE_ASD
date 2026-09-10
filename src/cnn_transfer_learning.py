import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class FHJOCNNTransfer(nn.Module):
    def __init__(self, k_pivotal=50):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        old_conv = self.backbone.features[0][0]
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=False)
        new_conv.weight.data = old_conv.weight.data[:, :1, :, :].clone()
        self.backbone.features[0][0] = new_conv
        
        in_f = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_f, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.jellyfish_mask = nn.Parameter(torch.ones(1, 1, k_pivotal, k_pivotal))

    def forward(self, x_2d):
        optimized = x_2d * torch.sigmoid(self.jellyfish_mask)
        x = F.interpolate(optimized, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x)