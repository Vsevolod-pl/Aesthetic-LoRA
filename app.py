import json
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageEnhance

import inference

UPLOAD_DIR = tempfile.mkdtemp(prefix="iqavlm_")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference.init_models("config.yaml")
    yield
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)


app = FastAPI(lifespan=lifespan)


def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename).suffix or ".jpg"
    fd, path = tempfile.mkstemp(suffix=suffix, dir=UPLOAD_DIR)
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return path


def _rotate_hue(img: Image.Image, degrees: float) -> Image.Image:
    arr = np.array(img, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    flat = arr.reshape(-1, 3)

    r, g, b = flat[:, 0], flat[:, 1], flat[:, 2]
    maxc = flat.max(axis=1)
    diff = maxc - flat.min(axis=1)

    v = maxc
    s = np.where(maxc > 1e-6, diff / maxc, 0.0)

    hue_arr = np.zeros(len(flat), dtype=np.float32)
    idx_r = (maxc == r) & (diff > 1e-6)
    idx_g = (maxc == g) & (diff > 1e-6)
    idx_b = (maxc == b) & (diff > 1e-6)
    hue_arr[idx_r] = ((g[idx_r] - b[idx_r]) / diff[idx_r]) % 6
    hue_arr[idx_g] = (b[idx_g] - r[idx_g]) / diff[idx_g] + 2
    hue_arr[idx_b] = (r[idx_b] - g[idx_b]) / diff[idx_b] + 4
    hue_arr = (hue_arr / 6.0 + degrees / 360.0) % 1.0

    hue6 = hue_arr * 6.0
    i = np.floor(hue6).astype(int)
    f = hue6 - np.floor(hue6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    result = np.zeros_like(flat)
    for sector, (ri, gi, bi) in enumerate(
        [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    ):
        mask = i % 6 == sector
        result[mask, 0] = ri[mask]
        result[mask, 1] = gi[mask]
        result[mask, 2] = bi[mask]

    out = np.clip(result * 255, 0, 255).astype(np.uint8).reshape(h, w, 3)
    return Image.fromarray(out)


def _apply_edits(src_path: str, params: dict) -> str:
    img = Image.open(src_path).convert("RGB")

    brightness = float(params.get("brightness", 1.0))
    contrast = float(params.get("contrast", 1.0))
    saturation = float(params.get("saturation", 1.0))
    temperature = float(params.get("temperature", 0))
    hue = float(params.get("hue", 0))
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    if temperature != 0:
        arr = np.array(img, dtype=np.float32)
        factor = temperature / 100.0
        arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 + factor * 0.2), 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - factor * 0.2), 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
    if hue != 0:
        img = _rotate_hue(img, hue)

    fd, path = tempfile.mkstemp(suffix=".jpg", dir=UPLOAD_DIR)
    os.close(fd)
    img.save(path, "JPEG", quality=92)
    return path


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/config")
async def get_config():
    return {
        "models": inference.get_model_choices(),
        "default_describe_prompt": inference.get_default_describe_prompt(),
        "mock_mode": inference.is_mock_mode(),
        "images_folder_enabled": bool(inference.get_images_folder()),
    }


@app.post("/api/compare")
async def compare(
    image1: UploadFile = File(...),
    image2: Optional[UploadFile] = File(default=None),
    edit_params: Optional[str] = Form(default=None),
    model_key: str = Form(...),
    describe_prompt: Optional[str] = Form(default=None),
):
    path1 = _save_upload(image1)
    path2 = None
    try:
        if image2 is not None and image2.filename:
            path2 = _save_upload(image2)
        elif edit_params:
            params = json.loads(edit_params)
            path2 = _apply_edits(path1, params)
        else:
            raise HTTPException(400, "Provide either image2 or edit_params")

        result = inference.run_inference(
            model_key=model_key,
            img1_path=path1,
            img2_path=path2,
            describe_prompt=describe_prompt or None,
        )
        return JSONResponse(content=result)
    finally:
        for p in (path1, path2):
            if p and os.path.exists(p):
                os.unlink(p)


@app.get("/api/images")
async def list_images():
    folder = inference.get_images_folder()
    if not folder:
        raise HTTPException(404, "Images folder not configured")
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise HTTPException(404, "Images folder not found")
    files = sorted(
        f for f in os.listdir(folder_path)
        if Path(f).suffix.lower() in _IMAGE_EXTS
    )
    return JSONResponse(content={"images": files})


@app.get("/api/images/{filename}")
async def serve_image(filename: str):
    folder = inference.get_images_folder()
    if not folder:
        raise HTTPException(404, "Images folder not configured")
    folder_path = Path(folder).resolve()
    file_path = (folder_path / filename).resolve()
    if not str(file_path).startswith(str(folder_path)):
        raise HTTPException(403, "Access denied")
    if not file_path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8032)
