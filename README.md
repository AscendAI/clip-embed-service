# CLIP Embed Service (Local API)

A tiny FastAPI service that takes an **image URL / data URI / raw base64 / local path** and returns a **normalized CLIP embedding**. Optional persistence to **Postgres + pgvector** (e.g., Neon).

> Runs CPU-only or with CUDA. Model is configurable via env.

---

## ✨ Endpoints
- `GET /health` – health & model info
- `POST /embed/image` – embed images (one or many)
- `POST /embed/text` – embed texts (one or many)

---

## 🧱 Requirements
- Python 3.10+
- (Optional) CUDA 12.1+ driver for GPU
- (Optional) Postgres 15+ with `pgvector` extension

Ubuntu prerequisites:
```bash
sudo apt-get update && sudo apt-get install -y build-essential curl ca-certificates
# clip-embed-service
