import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

import inference

UPLOAD_DIR = tempfile.mkdtemp(prefix="iqavlm_")


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


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/api/config")
async def get_config():
    return {
        "models": inference.get_model_choices(),
        "default_describe_prompt": inference.get_default_describe_prompt(),
    }


@app.post("/api/compare")
async def compare(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    model_key: str = Form(...),
    describe_prompt: str = Form(None),
):
    path1 = _save_upload(image1)
    path2 = _save_upload(image2)
    try:
        result = inference.run_inference(
            model_key=model_key,
            img1_path=path1,
            img2_path=path2,
            describe_prompt=describe_prompt or None,
        )
        return JSONResponse(content=result)
    finally:
        os.unlink(path1)
        os.unlink(path2)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8032)
