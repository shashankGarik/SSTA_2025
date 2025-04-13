import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        resnet = resnet18(ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(resnet.children())[:-5])

    def forward(self,x):
        return self.resnet(x)