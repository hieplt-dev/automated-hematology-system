import cv2
import torch

from src.ahs.models.faster_rcnn import BCCD_Model
from src.ahs.transforms.transforms_albu import build_val_aug_albu
from src.ahs.utils.load_config import load_config
from src.ahs.utils.visualize_img import visualize_img


def infer(img_path, checkpoint="experiments/outputs/best.pt", image_size=480, num_cls=3, score_thresh=0.5):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model & load weights
    model = BCCD_Model(num_classes=num_cls).model
    model.load_state_dict(torch.load(checkpoint, map_location=device)['model_state'])
    model.to(device)
    model.eval()

    # Build validation transform (resize + tensor)
    val_transform = build_val_aug_albu(image_size)

    # Read input image (RGB)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Albumentations expects dict
    transformed = val_transform(image=img_rgb, bboxes=[], labels=[])
    img_tensor = transformed["image"].float() / 255.0   # (C,H,W) [0,1]

    # Inference
    with torch.no_grad():
        prediction = model([img_tensor.to(device)])[0]

    # Visualize (only boxes with confidence > 0.5)
    keep = prediction["scores"] > score_thresh
    boxes  = prediction["boxes"][keep]
    labels = prediction["labels"][keep]

    visualize_img(img_tensor.cpu(), boxes, labels)
    
    return boxes, labels