import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
from utils import draw_keypoints_on_image
import torch.nn as nn
import torch.optim as optim
import torch
from lightning.pytorch.callbacks import Callback
import lightning as L

## Define Dataset
class CameraSequentialPairs(Dataset):
    def __init__(self, root, transform=None, min_offset=1, max_offset=5):
        """
        Args:
            root (str): Root directory containing camera folders.
            transform: Transformations to apply to each image.
            min_offset (int): Minimum frame difference between pairs.
            max_offset (int): Maximum frame difference between pairs.
        """
        self.transform = transform or transforms.ToTensor()
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.sequences = []

        # List all camera directories
        camera_dirs = sorted([os.path.join(root, d) for d in os.listdir(root)
                              if os.path.isdir(os.path.join(root, d))])
        
        # For each camera directory, get sorted image paths (using numeric sorting)
        for cam_dir in camera_dirs:
            image_dir = os.path.join(cam_dir, "images")
            if not os.path.exists(image_dir):
                continue
            image_files = sorted(
                [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')],
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])
            )
            if len(image_files) > 1:
                self.sequences.append(image_files)
        
        # Build cumulative lengths for indexing purposes
        self.cumulative_lengths = []
        total = 0
        for seq in self.sequences:
            # Each valid pair comes from a starting image (all except the last)
            total += len(seq) - 1
            self.cumulative_lengths.append(total)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
    # Find which sequence to use
        seq_idx = 0
        while idx >= self.cumulative_lengths[seq_idx]:
            seq_idx += 1
        seq = self.sequences[seq_idx]
        
        local_idx = idx if seq_idx == 0 else idx - self.cumulative_lengths[seq_idx - 1]
        
        # Check that the index is valid
        if local_idx >= len(seq) - 1:
            raise IndexError("Local index out of range for the sequence")
        
        img1_path = seq[local_idx]

        # Compute the maximum valid offset
        remaining = len(seq) - 1 - local_idx
        effective_max_offset = min(self.max_offset, remaining)
        
        # In case effective_max_offset < min_offset, use effective_max_offset
        # Otherwise, randomly sample
        offset = effective_max_offset  # or if you want random:
        # offset = random.randint(self.min_offset, effective_max_offset)

        img2_idx = local_idx + offset
        
        img2_path = seq[img2_idx]

        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
        except Exception as e:
            print(f"Error loading images: {img1_path} or {img2_path}: {e}")
            # Optionally, raise an error here or return a default dummy tensor
            raise e
        
        return self.transform(img1), self.transform(img2)


## camera dataset to dataloaders
def get_camera_dataloaders(data_root="./dataset_00", batch_size=8, min_offset=5, max_offset=30, num_workers=8):
    train_dataset = CameraSequentialPairs(
        root=os.path.join(data_root, "train"),
        transform=transforms.ToTensor(),
        min_offset=min_offset,
        max_offset=max_offset
    )
    
    val_dataset = CameraSequentialPairs(
        root=os.path.join(data_root, "val"),
        transform=transforms.ToTensor(),
        min_offset=min_offset,
        max_offset=max_offset
    )
    
    print("Training dataset pairs:", len(train_dataset))
    print("Validation dataset pairs:", len(val_dataset))
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader

class AlphaScheduler(Callback):
    def __init__(self, warmup_epochs=5, ramp_epochs=20, final_alpha=0.003):
        """
        alpha is the
        Args:
            warmup_epochs: Number of epochs to keep alpha = 0
            ramp_epochs: Number of epochs to linearly ramp up alpha after warmup
            final_alpha: Target alpha value
        """
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.final_alpha = final_alpha

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch

        if epoch < self.warmup_epochs:
            alpha = 0.0
        elif epoch < self.warmup_epochs + self.ramp_epochs:
            # Linear ramp from 0 to final_alpha
            ramp_progress = (epoch - self.warmup_epochs) / self.ramp_epochs
            alpha = ramp_progress * self.final_alpha
        else:
            alpha = self.final_alpha

        pl_module.alpha = alpha

class LitKeypointDetector(L.LightningModule):
    def __init__(self, keypoint_encoder, feature_encoder, feature_decoder):
        super().__init__()
        self.keypoint_generator = keypoint_encoder
        self.feature_encoder = feature_encoder
        self.image_reconstructor = feature_decoder
        self.alpha = 0

    def training_step(self, batch, batch_idx):
        v1, vt = batch
        
        v1_features = self.feature_encoder(v1)

        _, _, v1_heatmaps = self.keypoint_generator(v1)
        _, vt_soft_keypoints, vt_heatmaps = self.keypoint_generator(vt)

        vt_pred = self.image_reconstructor(v1_features, vt_soft_keypoints)
        loss, mse, condens = self.keypoint_loss(vt, vt_pred, v1_heatmaps, vt_heatmaps)
        self.log("train/total_loss", loss)
        self.log("train/mse", mse)
        self.log("train/condensation_loss", condens)
        self.log("alpha", self.alpha)

        return loss
    
    def validation_step(self, batch, batch_idx):
        v1, vt = batch

        # Encode features from v0
        v1_features = self.feature_encoder(v1)

        # Get keypoints from v0 and v1 separately
        _, soft_kp_v1, v1_heatmaps = self.keypoint_generator(v1)
        _, soft_kp_vt, vt_heatmaps = self.keypoint_generator(vt)

        # Predict v1 using v0 features + v1 keypoints
        vt_pred = self.image_reconstructor(v1_features, soft_kp_vt)
        loss, mse, condens = self.keypoint_loss(vt, vt_pred, v1_heatmaps, vt_heatmaps)

        self.log("val/total_loss", loss)
        self.log("val/mse", mse)
        self.log("val/condensation_loss", condens)

        if batch_idx == 0:
            B, C, img_H, img_W = v1.shape
            _, _, heatmap_H, heatmap_W = vt_heatmaps.shape

            scale_x = img_W / heatmap_W
            scale_y = img_H / heatmap_H

            # Rescale keypoints
            scaled_kps_v0 = soft_kp_v1[0].clone()
            scaled_kps_v0[..., 0] *= scale_x
            scaled_kps_v0[..., 1] *= scale_y

            scaled_kps_v1 = soft_kp_vt[0].clone()
            scaled_kps_v1[..., 0] *= scale_x
            scaled_kps_v1[..., 1] *= scale_y

            # Clamp images for visualization
            v0_img = v1[0].clamp(0, 1)
            v1_img = vt[0].clamp(0, 1)
            v1_pred_img = vt_pred[0].clamp(0, 1)

            # Overlay keypoints
            vis_v0 = draw_keypoints_on_image(v0_img, scaled_kps_v0, color='b')  # Blue keypoints
            vis_v1 = draw_keypoints_on_image(v1_img, scaled_kps_v1, color='g')  # Red keypoints
            vis_pred = draw_keypoints_on_image(v1_pred_img, scaled_kps_v1, color='g')

            # Combine visuals
            grid = make_grid([vis_v0, vis_v1, vis_pred])
            self.logger.experiment.add_image("val/visualize_key_reconstruction", grid, self.current_epoch)

            # Log heatmaps of v1
            sample_heatmaps = vt_heatmaps[0].unsqueeze(1)  # [K, 1, H, W]
            heatmap_grid = make_grid(sample_heatmaps, nrow=10, normalize=True, scale_each=True)
            self.logger.experiment.add_image("val/heatmaps_v1", heatmap_grid, self.current_epoch)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
    
    def keypoint_loss(self, vt, vt_pred, v0_heat, vt_heat):

        mse = nn.functional.mse_loss(vt, vt_pred)

        condensation_loss_1 = self.condensation_loss_entropy(v0_heat)
        condensation_loss_t = self.condensation_loss_entropy(vt_heat)
        condensation_loss = (condensation_loss_1 + condensation_loss_t)/2

        return mse+self.alpha*condensation_loss, mse, condensation_loss
    
    def condensation_loss_entropy(self, heatmaps):
        # heatmaps: [B, K, H, W] (softmaxed)
        eps = 1e-8
        log_h = torch.log(heatmaps + eps)
        entropy = -torch.sum(heatmaps * log_h, dim=(-2, -1))  # [B, K]
        return entropy.mean()



from models.keypoints.KeypointPredictor import KeypointPredictor
from models.keypoints.ImageEncoder import ImageEncoder
from models.keypoints.ImageDecoder import ImageDecoder

if __name__ == "__main__":

    train_loader, val_loader = get_camera_dataloaders(
    data_root="./dataset_00", batch_size=16, min_offset=5, max_offset=30, num_workers=8
)
    
    trainer = L.Trainer(max_epochs=200, callbacks=[AlphaScheduler()])
    model = LitKeypointDetector(KeypointPredictor(100), ImageEncoder(), ImageDecoder(64, 100))

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
