import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image
import io

def draw_keypoints_on_image(image_tensor, keypoints, color='r'):
    """
    image_tensor: [3, H, W] (float in [0,1])
    keypoints: [N, 2] with (x, y) format in image coordinates
    """
    image = TF.to_pil_image(image_tensor.cpu())
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.axis("off")
    for x, y in keypoints.cpu():
        plt.scatter(x, y, c=color, s=10)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    pil_image = Image.open(buf)
    return TF.to_tensor(pil_image)