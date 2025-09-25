import io, os, base64, json, logging
from typing import List, Optional

import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import open_clip
import torch
import torch.nn.functional as F

# Optional DB
from pgvector.psycopg import register_vector
import psycopg

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("clip-embed-service")

# ==== Config ====
MODEL_NAME = os.getenv("MODEL_NAME", "ViT-B-32")
PRETRAINED = os.getenv("PRETRAINED", "laion2b_s34b_b79k")
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))
MAX_BATCH = int(os.getenv("MAX_BATCH", "64"))
NEON_URL = os.getenv("NEON_URL", "").strip()
TABLE_NAME = os.getenv("TABLE_NAME", "vectors")

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

# ==== Optional DB bootstrap ====
conn = None
if NEON_URL:
    try:
        conn = psycopg.connect(NEON_URL, autocommit=True)
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                source TEXT NOT NULL,
                emb VECTOR({_EMB_DIM}) NOT NULL,
                meta JSONB DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ DEFAULT now()
            );
            """)
            # Verify existing vector dimension; alter if mismatched
            cur.execute(
                "SELECT atttypmod-4 FROM pg_attribute "
                "JOIN pg_class ON pg_class.oid=attrelid "
                "WHERE relname=%s AND attname='emb';",
                (TABLE_NAME,),
            )
            row = cur.fetchone()
            if row and row[0] != _EMB_DIM:
                log.warning(f"Vector dim mismatch {row[0]} != {_EMB_DIM}; altering column...")
                cur.execute(f"ALTER TABLE {TABLE_NAME} ALTER COLUMN emb TYPE vector({_EMB_DIM});")
        log.info("DB ready.")
    except Exception as e:
        log.error(f"DB init failed: {e}")
        conn = None

# ==== API ====
app = FastAPI(title="CLIP Embed Service", version="1.0.0")

class EmbedImageRequest(BaseModel):
    inputs: List[str] = Field(..., min_items=1, max_items=MAX_BATCH)
    persist: bool = False
    meta: Optional[dict] = None

class EmbedTextRequest(BaseModel):
    inputs: List[str] = Field(..., min_items=1, max_items=MAX_BATCH)

def _persist_rows(sources: List[str], embs: np.ndarray, meta: Optional[dict] = None) -> int:
    if not conn:
        return 0
    m = meta or {}
    rows = [(src, emb.tolist(), json.dumps(m)) for src, emb in zip(sources, embs)]
    q = f"INSERT INTO {TABLE_NAME} (source, emb, meta) VALUES (%s, %s, %s)"
    with conn.cursor() as cur:
        for src, emb_list, meta_json in rows:
            cur.execute(q, (src, emb_list, meta_json))
    return len(rows)

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": device,
        "model": MODEL_NAME,
        "pretrained": PRETRAINED,
        "dim": _EMB_DIM,
        "db": bool(conn),
    }

@app.post("/embed/image")
def embed_image(req: EmbedImageRequest):
    try:
        imgs = [_load_image(s) for s in req.inputs]
        embs = embed_images(imgs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    saved = 0
    if req.persist:
        try:
            saved = _persist_rows(req.inputs, embs, req.meta)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Persist failed: {e}")
    return {
        "dim": int(embs.shape[-1]),
        "model": MODEL_NAME,
        "pretrained": PRETRAINED,
        "count": int(embs.shape[0]),
        "embeddings": embs.tolist(),
        "saved": saved
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
