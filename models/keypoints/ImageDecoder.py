import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ImageDecoder(nn.Module):
    def __init__(self, feature_channels, num_keypoints, output_channels=3, heatmap_dims = (32,32), condense = False):
        """
        Args:
         - feature_channels: number of channels in the encoder's feature maps.
         - num_keypoints: number of keypoint heatmaps.
         - output_channels: number of channels in the reconstructed output (e.g., 3 for RGB).
        """
        super(ImageDecoder, self).__init__()
        
        # The input to the decoder is the concatenation of feature maps and heatmaps.
        input_channels = feature_channels + num_keypoints
        self.num_keypoints = num_keypoints
        # Initial convolution layers to fuse the information.
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Transpose convolution (deconvolution) blocks for upsampling.
        # Adjust kernel_size, stride, and padding as needed based on your target output size.
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Upsample x2
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # Upsample x2 again
        self.deconv3 = nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1)  # Final upsample

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)  # conv (to smoothen)
        self.conv5 = nn.Conv2d(32, output_channels, kernel_size=3, stride=1, padding=1)  # final conv (to smoothen)

        self.act = nn.SiLU()
        self.tanh = nn.Tanh()  # Constrain output between -1 and 1 (for normalized images)

        H,W = heatmap_dims
        y_map = torch.arange(0, H).view(1, 1, H, 1).float()
        x_map = torch.arange(0, W).view(1, 1, 1, W).float()
        self.register_buffer("x_map", x_map)
        self.register_buffer("y_map", y_map)
        self.condense = condense

        self.layernorm = nn.InstanceNorm2d(32)


    def forward(self, feature_maps, keypoints, return_heatmaps=False, return_t2nod=False):
        """
        Args:
         - feature_maps: [B, C, H, W] tensor from the encoder.
         - keypoints: [B, N, 2] tensor in heatmap space
        Returns:
         - reconstructed_image: [B, output_channels, H_out, W_out] tensor.
        """
        # Concatenate along channel dimension

        heatmaps = self.render_gaussian_heatmaps(keypoints)
        if self.condense:
            heatmaps = self.logsumexp_pooling(heatmaps)
        #     feature_maps = torch.zeros_like(feature_maps)
        x = torch.cat([feature_maps, heatmaps], dim=1)  # Shape: [B, C + N, H, W]

        # Fuse the features with standard convolutions
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))

        # Upsample using transpose convolutions
        x = self.act(self.deconv1(x))
        x = self.act(self.deconv2(x))
        x = self.act(self.deconv3(x))  
        x = self.layernorm(x)

        x = self.act(self.conv4(x))
        x = self.conv5(x)

        if return_t2nod:
            logits = x[:, 0:1]                # no activation (raw logits for BCE)
            t2no   = self.tanh(x[:, 1:2])      # for normalized regression
            t2nd   = self.tanh(x[:, 2:3])

            x = torch.cat([logits, t2no, t2nd], dim=1)
        else:
            x = self.tanh(x)

        if return_heatmaps:
            return x, heatmaps
        return x
    
    def render_gaussian_heatmaps(self, keypoints, sigma=2.0):
        B, N, _ = keypoints.shape

        # Split keypoints
        x_k = keypoints[..., 0].view(B, N, 1, 1)
        y_k = keypoints[..., 1].view(B, N, 1, 1)

        # Compute squared distance from keypoint
        dist_sq = (self.x_map - x_k) ** 2 + (self.y_map - y_k) ** 2
        heatmaps = torch.exp(-dist_sq / (2 * sigma ** 2))

        return heatmaps
    
    def logsumexp_pooling(self, heatmaps, beta=10.0):
        """
        Consolidate heatmaps using LogSumExp pooling.

        Args:
            heatmaps: Tensor of shape (B, K, H, W) where K = num_keypoints
            num_groups: Number of final heatmaps desired
            beta: Temperature parameter for LogSumExp

        Returns:
            Tensor of shape (B, num_groups, H, W)
        """
        B, K, H, W = heatmaps.shape
        assert K//self.num_keypoints
        group_size = K//self.num_keypoints

        # Reshape: (B, num_groups, group_size, H, W)
        grouped = heatmaps.view(B, self.num_keypoints, group_size, H, W)

        # LogSumExp pooling over the group dimension
        pooled = (1.0 / beta) * torch.logsumexp(beta * grouped, dim=2)  # shape: (B, num_groups, H, W)

        return pooled