# app.py - FastAPI backend for TB prediction + GradCAM + LIME
import io
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
CLASS_MAP = {0: "Normal", 1: "Tuberculosis"}


# ---------------- MODEL ----------------
def build_model(num_classes=2):
    model = models.densenet121(weights=None)       # No pretrained weights
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )
    return model


def load_model(model_path=MODEL_PATH, device=DEVICE):
    model = build_model(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)

    # Support both "model_state_dict" and raw state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ---------------- PREPROCESSING ----------------
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- FASTAPI ----------------
app = FastAPI(title="TB X-ray Classifier + Explainability API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev (React)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
try:
    model = load_model()
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Failed to load model:", e)
    model = None


# ---------------- RESPONSE MODEL ----------------
class PredictionResponse(BaseModel):
    prediction: str
    label_index: int
    confidence: float


@app.get("/")
def root():
    return {"status": "ok", "message": "TB prediction API. POST image to /predict"}


# ---------------- PREDICT ENDPOINT ----------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")

    input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    label_idx = int(probs.argmax())
    confidence = float(probs[label_idx])

    return PredictionResponse(
        prediction=CLASS_MAP[label_idx],
        label_index=label_idx,
        confidence=round(confidence, 4)
    )


# =================================================================
# -----------------------  GRAD-CAM  ------------------------------
# =================================================================
import cv2
import numpy as np

def find_last_conv(model):
    last = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last = (name, m)
    return last[0]


class GradCAM:
    def __init__(self, model, device):
        self.model = model.eval()
        self.device = device

        self.gradients = None
        self.activations = None

        target_layer = find_last_conv(model)

        # Hook activations
        def forward_hook(_, __, output):
            self.activations = output

        # Hook gradients
        def backward_hook(_, grad_input, grad_output):
            self.gradients = grad_output[0]

        for name, module in self.model.named_modules():
            if name == target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break

    def generate(self, image_tensor):
        output = self.model(image_tensor)
        class_idx = output.argmax().item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        weights = gradients.mean(axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam, class_idx


gradcam = GradCAM(model, DEVICE)


def overlay_cam(pil_img, cam):
    img = np.array(pil_img)
    h, w = img.shape[:2]
    cam = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, 0.45, img[:, :, ::-1], 0.55, 0)
    overlay = overlay[:, :, ::-1]
    return Image.fromarray(overlay)


@app.post("/gradcam")
async def gradcam_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    cam, class_idx = gradcam.generate(tensor)
    heatmap = overlay_cam(img, cam)

    buf = io.BytesIO()
    heatmap.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


# =================================================================
# -------------------------- LIME --------------------------------
# =================================================================
from lime import lime_image
from skimage.segmentation import mark_boundaries


def batch_predict(np_imgs):
    tensors = []
    for arr in np_imgs:
        pil = Image.fromarray(arr)
        tensors.append(preprocess(pil).unsqueeze(0))
    batch = torch.cat(tensors).to(DEVICE)

    with torch.no_grad():
        out = model(batch)
        probs = torch.softmax(out, dim=1).cpu().numpy()
    return probs


@app.post("/explain")
async def explain_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    img_np = np.array(img)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np,
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=600   # reduce for speed
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    lime_img = mark_boundaries(temp / 255.0, mask)
    lime_img = (lime_img * 255).astype(np.uint8)
    lime_img = Image.fromarray(lime_img)

    buf = io.BytesIO()
    lime_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
