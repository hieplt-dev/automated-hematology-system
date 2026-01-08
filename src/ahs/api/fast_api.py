from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import torch
import os

from src.ahs.models.faster_rcnn import BCCD_Model
from src.ahs.transforms.transforms_albu import build_val_aug_albu

# Config
checkpoint_path = "/models/best_qint8.pt"
image_size = 480
score_thresh = 0.5  # default score threshold

app = FastAPI(title="AHS Inference API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoint exists
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

# load model on startup
model = None


@app.on_event("startup")
def load_model():
    global model
    m = torch.load(checkpoint_path, map_location=device, weights_only=False)
    m.to(device)
    m.eval()
    model = m


# Build validation transform (resize + tensor)
val_transform = build_val_aug_albu(image_size)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")  # remove response_model to avoid schema mismatch issues here
async def predict(file: UploadFile = File(...), score_thresh: float = 0.5):
    contents = await file.read()
    try:
        # Try to decode image bytes (accept variety of content-types)
        arr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cannot decode image bytes")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Albumentations expects dict
        transformed = val_transform(image=img_rgb, bboxes=[], labels=[])
        img_tensor = transformed["image"].float() / 255.0  # (C,H,W) [0,1]

        # Inference (don't block event loop in real app; use executor)
        with torch.no_grad():
            prediction = model([img_tensor.to(device)])[0]

        # Filter and convert to python lists
        scores_t = prediction["scores"]
        keep = scores_t > float(score_thresh)
        boxes_t = prediction["boxes"][keep]
        labels_t = prediction["labels"][keep]
        scores_t = scores_t[keep]

        boxes = boxes_t.cpu().numpy().tolist() if boxes_t.numel() else []
        labels = labels_t.cpu().numpy().astype(int).tolist() if labels_t.numel() else []
        scores = (
            scores_t.cpu().numpy().astype(float).tolist() if scores_t.numel() else []
        )

        return {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "meta": {"num_boxes": len(labels)},
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        # don't leak internal trace in production
        raise HTTPException(status_code=500, detail="inference failed")
