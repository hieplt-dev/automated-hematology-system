import os
import time

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from src.ahs.models.faster_rcnn import BCCD_Model
from src.ahs.transforms.transforms_albu import build_val_aug_albu
from src.ahs.api.storage import StorageClient

# Prometheus metrics
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 5),
)

MODEL_INFERENCE_DURATION = Histogram(
    "model_inference_duration_seconds",
    "Model inference latency",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1, 2),
)

IN_FLIGHT_REQUESTS = Gauge(
    "http_requests_in_flight",
    "Number of in-flight HTTP requests",
)

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

# Initialize Storage Client
storage_client = StorageClient()

# Mount static files
from fastapi.staticfiles import StaticFiles
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

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


# expose metrics endpoint
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
async def predict(file: UploadFile = File(...), score_thresh: float = 0.5):
    # Record request start time for end-to-end latency measurement
    start_time = time.time()

    # Increase gauge to indicate one more in-flight request
    IN_FLIGHT_REQUESTS.inc()

    try:
        # Read uploaded file bytes (image)
        contents = await file.read()

        # Decode image bytes into OpenCV format
        arr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("cannot decode image bytes")

        # Convert BGR (OpenCV default) to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Apply validation transforms (resize, normalize, to tensor)
        transformed = val_transform(image=img_rgb, bboxes=[], labels=[])
        img_tensor = transformed["image"].float() / 255.0  # (C, H, W), [0, 1]

        # ============================
        # Model inference timing
        # ============================
        infer_start = time.time()
        with torch.no_grad():
            prediction = model([img_tensor.to(device)])[0]
        MODEL_INFERENCE_DURATION.observe(time.time() - infer_start)

        # Extract prediction outputs
        scores_t = prediction["scores"]

        # Filter detections by confidence threshold
        keep = scores_t > float(score_thresh)
        boxes_t = prediction["boxes"][keep]
        labels_t = prediction["labels"][keep]
        scores_t = scores_t[keep]

        # Convert tensors to Python-native types for JSON response
        boxes = boxes_t.cpu().numpy().tolist() if boxes_t.numel() else []
        labels = labels_t.cpu().numpy().astype(int).tolist() if labels_t.numel() else []
        scores = (
            scores_t.cpu().numpy().astype(float).tolist() if scores_t.numel() else []
        )

        # Increment successful request counter
        HTTP_REQUESTS_TOTAL.labels("POST", "/predict", "200").inc()

        # Return inference result
        result = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "meta": {"num_boxes": len(labels)},
        }

        # Save to storage if available
        if storage_client.s3:
            timestamp = int(time.time())
            filename = f"{timestamp}_{file.filename}"
            
            # Reset file pointer to beginning to read bytes again for upload, 
            # or just use the bytes we already read
            # content is already in variable `contents`
            
            image_key = storage_client.upload_image(contents, filename)
            
            if image_key:
                result["image_key"] = image_key
                # Save full result with image reference
                storage_client.upload_json(result, f"{timestamp}.json")

        return result

    except ValueError as e:
        # Client-side error (invalid input image)
        HTTP_REQUESTS_TOTAL.labels("POST", "/predict", "400").inc()
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        # Server-side error (model failure, runtime error, etc.)
        HTTP_REQUESTS_TOTAL.labels("POST", "/predict", "500").inc()
        raise HTTPException(status_code=500, detail="inference failed")

    finally:
        # Observe total request latency (end-to-end)
        HTTP_REQUEST_DURATION.labels("POST", "/predict").observe(
            time.time() - start_time
        )

        # Decrease in-flight request gauge
        IN_FLIGHT_REQUESTS.dec()


@app.get("/history")
async def get_history(limit: int = 50):
    if not storage_client.s3:
        return []
    return storage_client.list_history(limit=limit)


@app.get("/images/{key:path}")
async def get_image(key: str):
    if not storage_client.s3:
        raise HTTPException(status_code=404, detail="Storage not available")
    
    content = storage_client.get_file_content(key)
    if content is None:
        raise HTTPException(status_code=404, detail="Image not found")
        
    return Response(content, media_type="image/png")
