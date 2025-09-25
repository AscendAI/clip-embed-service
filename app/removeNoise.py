from __future__ import annotations

import io, base64, time, requests
from typing import List, Tuple, Optional
from PIL import Image, ImageFile
import numpy as np
import cv2

# Pillow safety for large/partial images
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Basic HTTP headers to avoid some CDNs blocking requests
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

def http_get(url: str, timeout: int = 30, retries: int = 3, backoff: float = 0.8) -> bytes:
    """Fetch URL with retries; raises on failure."""
    err = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r.content
        except Exception as e:
            err = e
            time.sleep(backoff * (2**i))
    raise RuntimeError(f"HTTP fetch failed: {err}")

def to_rgb(img: Image.Image) -> Image.Image:
    """Convert to RGB, flattening alpha if present."""
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img.convert("RGB"), mask=img.split()[-1])
        return bg
    return img.convert("RGB") if img.mode != "RGB" else img

def load_image(inp: str | bytes | Image.Image) -> Image.Image:
    """
    Load from:
      - URL (http/https)
      - data: URI base64
      - raw base64 str
      - bytes
      - local path
      - already-opened PIL.Image
    """
    try:
        if isinstance(inp, Image.Image):
            return to_rgb(inp)

        if isinstance(inp, bytes):
            return to_rgb(Image.open(io.BytesIO(inp)))

        if isinstance(inp, str):
            s = inp.strip()
            if s.startswith(("http://", "https://")):
                return to_rgb(Image.open(io.BytesIO(http_get(s))))
            if s.startswith("data:") and ";base64," in s:
                b64 = s.split(";base64,")[1]
                return to_rgb(Image.open(io.BytesIO(base64.b64decode(b64))))
            # heuristic: looks like base64 (safe quick check)
            charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r"
            if len(s) >= 16 and all(c in charset for c in s[:50]):
                return to_rgb(Image.open(io.BytesIO(base64.b64decode(s))))
            # else assume local path
            return to_rgb(Image.open(s))

        raise ValueError("Unsupported input type")
    except Exception as e:
        raise RuntimeError(f"Failed to load image: {e}")
def trim_solid_bars(rgb: np.ndarray, std_thresh: float = 2.0, max_trim_frac: float = 0.22
                   ) -> Tuple[np.ndarray, Tuple[int,int,int,int]]:
    """
    Remove uniform UI bars from top/bottom/left/right using low standard deviation.
    Limits trimming to ~22% per side to avoid over-cropping.
    Returns (trimmed_rgb, (left, top, right, bottom) in original coords).
    """
    h, w, _ = rgb.shape
    max_top = int(h * max_trim_frac)
    max_bot = h - max_top
    max_left = int(w * max_trim_frac)
    max_right = w - max_left

    # Per-row / per-column std
    row_std = rgb.reshape(h, -1).std(axis=1)
    col_std = rgb.transpose(1,0,2).reshape(w, -1).std(axis=1)

    top = 0
    while top < max_top and row_std[top] < std_thresh:
        top += 1
    bottom = h - 1
    while bottom > max_bot and row_std[bottom] < std_thresh:
        bottom -= 1

    left = 0
    while left < max_left and col_std[left] < std_thresh:
        left += 1
    right = w - 1
    while right > max_right and col_std[right] < std_thresh:
        right -= 1

    # clamp & ensure valid box
    top = max(0, min(top, h-2))
    left = max(0, min(left, w-2))
    bottom = max(top+1, min(bottom, h-1))
    right  = max(left+1, min(right,  w-1))
    return rgb[top:bottom+1, left:right+1, :], (left, top, right, bottom)

def largest_content_rect(rgb_trimmed: np.ndarray
                        ) -> Optional[Tuple[int,int,int,int]]:
    """
    Find the largest plausible rectangular content area via edges + contours.
    Returns (x,y,w,h) in TRIMMED coordinates, or None if not found.
    """
    th, tw, _ = rgb_trimmed.shape
    # Work on a downscaled copy for speed/robustness
    max_side = max(th, tw)
    scale = 800 / max_side
    if scale < 1.0:
        small = cv2.resize(rgb_trimmed, (int(tw*scale), int(th*scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = rgb_trimmed.copy()
        scale = 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best = None
    best_score = -1.0
    img_area = small.shape[0] * small.shape[1]

    for c in cnts:
        x,y,wc,hc = cv2.boundingRect(c)
        area = wc * hc
        if area < 0.14 * img_area:      # ignore tiny blocks
            continue
        aspect = wc / max(hc, 1)
        if not (0.4 <= aspect <= 2.4):  # photo-ish rectangle range
            continue
        # favor fuller regions with edges inside
        rect = edges[y:y+hc, x:x+wc]
        fill_ratio = rect.mean() / 255.0
        score = area * (0.6 + 0.4 * fill_ratio)
        if score > best_score:
            best_score = score
            best = (x, y, wc, hc)

    if best is None:
        return None

    x,y,wc,hc = best
    if scale != 1.0:
        x = int(x/scale); y = int(y/scale)
        wc = int(wc/scale); hc = int(hc/scale)
    return (x, y, wc, hc)

def extract_main_image(pil_img: Image.Image, pad: int = 8) -> Image.Image:
    """
    1) Trim uniform bars
    2) Find largest content rectangle
    3) Pad and crop
    Fallbacks: trimmed or original if nothing useful found.
    """
    rgb = np.array(pil_img)
    trimmed, (l0,t0,r0,b0) = trim_solid_bars(rgb)

    rect = largest_content_rect(trimmed)
    if rect is None:
        # fallback to trimmed if trimming changed area; else original
        crop = trimmed if trimmed.shape[:2] != rgb.shape[:2] else rgb
    else:
        x,y,w,h = rect
        x1 = max(0, x - pad); y1 = max(0, y - pad)
        x2 = min(trimmed.shape[1], x + w + pad)
        y2 = min(trimmed.shape[0], y + h + pad)
        crop = trimmed[y1:y2, x1:x2, :]

    return Image.fromarray(crop)

def run_remove_noise_on_url(url: str, save_as: str = None):
    """
    Fetch image by URL, process it to remove noise, and either save to disk or return the processed image.
    
    Args:
        url: The URL of the image to process
        save_as: Optional path to save the image. If None, the image is not saved to disk.
        
    Returns:
        PIL.Image.Image: The processed image object
    """
    img = load_image(url)
    cropped = extract_main_image(img)
    print("→ Cropped size:", cropped.size)
    
    # Save to disk only if save_as is provided
    if save_as:
        cropped.save(save_as)
        print(f"Saved: {save_as}")
        
    return cropped
# Paste any screenshot URL (Facebook/IG/Twitter/Shop link etc.)



test_url = "https://cdn.discordapp.com/attachments/1093165395466268743/1420669832843497596/image.png?ex=68d63d62&is=68d4ebe2&hm=aa65b63abb273efc987fb32c5951daf44590d774d226d40fcfd2d443c39e74a3&"

test_url1 = "https://cdn.discordapp.com/attachments/1093165395466268743/1420668169030467584/Screenshot_20250925-130753.png?ex=68d63bd6&is=68d4ea56&hm=e71666bf05808fc3eac1ccf69dffc46fd7a55039f9b877738d2eed7264320dc1&"

# run_remove_noise_on_url(test_url, "cropped_from_url.png")
# run_remove_noise_on_url(test_url1, "cropped_from_url.png")