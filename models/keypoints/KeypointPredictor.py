import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class KeypointPredictor(nn.Module):
    def __init__(self, num_keypoints = 100, heatmap_dims = (32,32), temperature=1): # N is the number of keypoints
        super(KeypointPredictor, self).__init__()
        resnet = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-5])

        self.heatmap_head = nn.Conv2d(in_channels=64, out_channels=num_keypoints, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        H, W = heatmap_dims

        x_map = torch.arange(W).view(1,1,1,W).float()
        y_map = torch.arange(H).view(1,1,H,1).float()

        self.register_buffer("x_map", x_map)
        self.register_buffer("y_map", y_map)

        self.temperature = temperature

    def compute_spatial_expectation(self, heatmaps):
        """
        Takes in output of the keypoint encoder and returns N kepoints
        inputs:
        - heatmaps [B, N, H, W]
        """
        total = torch.sum((heatmaps), dim =(-2,-1)) + 1e-6

        x_ = torch.sum(heatmaps*self.x_map, dim=(-2,-1))
        y_ = torch.sum(heatmaps*self.y_map, dim=(-2,-1))

        return torch.stack([x_/total, y_/total], dim = -1)

    def forward(self, x):

        x = self.resnet(x)
        x = self.heatmap_head(x)

        # x = self.sigmoid(x)

        B, K, H, W = x.shape
        heatmaps = x.view(B, K, -1) / self.temperature
        heatmaps = torch.softmax(heatmaps, dim=-1)
        heatmaps = heatmaps.view(B, K, H, W)

        soft_keypoints = self.compute_spatial_expectation(heatmaps)

        delta = (torch.round(soft_keypoints) - soft_keypoints).detach()
        
        return soft_keypoints + delta, soft_keypoints, heatmaps
