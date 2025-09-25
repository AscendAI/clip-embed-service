import io, os, base64, logging
from typing import List
import uuid

import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import open_clip
import torch
import torch.nn.functional as F

# Import removeNoise module for image noise removal
from app.removeNoise import run_remove_noise_on_url, extract_main_image

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("clip-embed-service")

# ==== Config ====
MODEL_NAME = os.getenv("MODEL_NAME", "ViT-B-32")
PRETRAINED = os.getenv("PRETRAINED", "laion2b_s34b_b79k")
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "64"))

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== Model ====
log.info(f"Loading OpenCLIP: {MODEL_NAME} / {PRETRAINED} on {device}")
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name=MODEL_NAME, pretrained=PRETRAINED
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model = model.to(device).eval()

@torch.inference_mode()
def embed_texts(texts: List[str]) -> np.ndarray:
    toks = tokenizer(texts).to(device)
    feats = model.encode_text(toks)
    feats = F.normalize(feats, dim=-1)
    return feats.detach().cpu().numpy()

@torch.inference_mode()
def embed_images(imgs: List[Image.Image]) -> np.ndarray:
    batch = torch.stack([preprocess(im) for im in imgs]).to(device)
    feats = model.encode_image(batch)
    feats = F.normalize(feats, dim=-1)
    return feats.detach().cpu().numpy()

def _load_image(inp: str) -> Image.Image:
    """
    Accepts:
      - http(s) URL
      - data URI (data:image/png;base64,....)
      - raw base64 (heuristic decode)
      - local path (/path/to/file.png)
    """
    try:
        if inp.startswith(("http://", "https://")):
            headers = {"User-Agent": "clip-embed-service/1.0"}
            r = requests.get(inp, headers=headers, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            return Image.open(io.BytesIO(r.content)).convert("RGB")
        if inp.startswith("data:") and ";base64," in inp:
            b64 = inp.split(";base64,")[1]
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        # heuristic for raw base64
        try:
            raw = base64.b64decode(inp, validate=True)
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            pass
        # local path
        with open(inp, "rb") as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")

# Determine embedding dim once at startup
_EMB_DIM = int(embed_texts(["probe"]).shape[-1])

# ==== API ====
app = FastAPI(title="CLIP Embed Service", version="1.0.0")

class EmbedImageRequest(BaseModel):
    inputs: List[str] = Field(..., min_items=1, max_items=MAX_BATCH)

class EmbedTextRequest(BaseModel):
    inputs: List[str] = Field(..., min_items=1, max_items=MAX_BATCH)

class RemoveNoiseRequest(BaseModel):
    url: str = Field(..., description="URL of the image to process")
    output_filename: str = Field(None, description="Optional custom filename for the output image")

class RemoveNoiseResponse(BaseModel):
    original_url: str
    output_path: str
    cropped_size: List[int]
    success: bool

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": device,
        "model": MODEL_NAME,
        "pretrained": PRETRAINED,
        "dim": _EMB_DIM,
    }

@app.post("/embed/image")
def embed_image(req: EmbedImageRequest):
    try:
        imgs = [_load_image(s) for s in req.inputs]
        embs = embed_images(imgs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        "dim": int(embs.shape[-1]),
        "model": MODEL_NAME,
        "pretrained": PRETRAINED,
        "count": int(embs.shape[0]),
        "embeddings": embs.tolist(),
    }

@app.post("/embed/text")
def embed_text(req: EmbedTextRequest):
    try:
        embs = embed_texts(req.inputs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "dim": int(embs.shape[-1]),
        "model": MODEL_NAME,
        "pretrained": PRETRAINED,
        "count": int(embs.shape[0]),
        "embeddings": embs.tolist(),
    }

@app.post("/remove-noise", response_model=RemoveNoiseResponse)
def remove_noise(req: RemoveNoiseRequest):
    try:
        # Generate unique filename if not provided
        output_filename = req.output_filename
        if not output_filename:
            output_filename = f"cropped_{uuid.uuid4().hex[:8]}.png"
        
        # Ensure the output path is in the current directory
        output_path = os.path.join(os.getcwd(), output_filename)
        
        # Process the image
        run_remove_noise_on_url(req.url, output_path)
        
        # Get the cropped image size
        img = Image.open(output_path)
        
        return {
            "original_url": req.url,
            "output_path": output_path,
            "cropped_size": list(img.size),
            "success": True
        }
    except Exception as e:
        log.error(f"Error processing image: {e}")
        raise HTTPException(status_code=400, detail=str(e))
