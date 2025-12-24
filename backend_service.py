from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional
import base64
import io
import os
import shutil
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(ROOT_DIR, "tmp")
UPLOAD_DIR = os.path.join(TMP_DIR, "uploads")
PREPROCESS_DIR = os.path.join(TMP_DIR, "preprocess")
PREVIEW_DIR = os.path.join(TMP_DIR, "previews")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREPROCESS_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

app = FastAPI()
app.mount("/files", StaticFiles(directory=TMP_DIR), name="files")


class PreprocessRequest(BaseModel):
    image_id: str
    mode: str = "auto"


class TaskRequest(BaseModel):
    image_id: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None


class TaskStatus(BaseModel):
    task_id: str
    status: str


class TaskResult(BaseModel):
    image_id: Optional[str] = None
    image_url: Optional[str] = None
    image_base64: Optional[str] = None


class TaskRecord(BaseModel):
    task_id: str
    status: str
    image_id: Optional[str]
    image_url: Optional[str]


IMAGES: dict[str, dict[str, str]] = {}
TASKS: dict[str, TaskRecord] = {}


def _expires_at(hours: int = 24) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()


def _make_image_url(subpath: str) -> str:
    return f"/files/{subpath}"


def _save_image_bytes(image_bytes: bytes, directory: str) -> tuple[str, str, str]:
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGBA")
    image_id = uuid.uuid4().hex
    filename = f"{image_id}.png"
    file_path = os.path.join(directory, filename)
    image.save(file_path, format="PNG")
    subpath = os.path.relpath(file_path, TMP_DIR)
    return image_id, file_path, _make_image_url(subpath)


def _load_image_path(image_id: str) -> str:
    record = IMAGES.get(image_id)
    if not record:
        raise HTTPException(status_code=404, detail="image_id not found")
    return record["path"]


def _decode_base64_image(payload: str) -> bytes:
    if "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


@app.post("/uploads")
async def upload_image(
    file: UploadFile = File(...),
    content_type: Optional[str] = Form(default=None),
) -> dict[str, Any]:
    data = await file.read()
    image_id, file_path, image_url = _save_image_bytes(data, UPLOAD_DIR)
    IMAGES[image_id] = {
        "path": file_path,
        "url": image_url,
        "expires_at": _expires_at(),
        "content_type": content_type or file.content_type or "image/png",
    }
    return {
        "image_id": image_id,
        "image_url": image_url,
        "expires_at": IMAGES[image_id]["expires_at"],
    }


@app.post("/preprocess")
async def preprocess_image(payload: PreprocessRequest) -> dict[str, str]:
    source_path = _load_image_path(payload.image_id)
    with open(source_path, "rb") as handle:
        data = handle.read()
    image_id, file_path, image_url = _save_image_bytes(data, PREPROCESS_DIR)
    IMAGES[image_id] = {
        "path": file_path,
        "url": image_url,
        "expires_at": _expires_at(),
        "content_type": "image/png",
    }
    return {"image_id": image_id, "image_url": image_url}


@app.post("/tasks")
async def create_task(payload: TaskRequest) -> dict[str, str]:
    image_id = None
    image_url = None

    if payload.image_id:
        if payload.image_id not in IMAGES:
            raise HTTPException(status_code=404, detail="image_id not found")
        image_id = payload.image_id
        image_url = IMAGES[payload.image_id]["url"]
    elif payload.image_url:
        image_url = payload.image_url
    elif payload.image_base64:
        data = _decode_base64_image(payload.image_base64)
        image_id, file_path, image_url = _save_image_bytes(data, UPLOAD_DIR)
        IMAGES[image_id] = {
            "path": file_path,
            "url": image_url,
            "expires_at": _expires_at(),
            "content_type": "image/png",
        }
    else:
        raise HTTPException(status_code=400, detail="image_id or image_url required")

    task_id = uuid.uuid4().hex
    TASKS[task_id] = TaskRecord(
        task_id=task_id,
        status="completed",
        image_id=image_id,
        image_url=image_url,
    )
    return {"task_id": task_id}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> TaskStatus:
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return TaskStatus(task_id=task_id, status=task.status)


def _make_preview(image_path: str, size: int) -> tuple[str, str]:
    base_name = os.path.basename(image_path)
    preview_name = f"preview_{size}_{base_name}"
    preview_path = os.path.join(PREVIEW_DIR, preview_name)
    if not os.path.exists(preview_path):
        image = Image.open(image_path)
        image.thumbnail((size, size))
        image.save(preview_path, format="PNG")
    subpath = os.path.relpath(preview_path, TMP_DIR)
    return preview_path, _make_image_url(subpath)


@app.get("/tasks/{task_id}/result")
async def get_task_result(
    task_id: str,
    thumbnail_only: bool = False,
    preview_size: Optional[int] = None,
) -> TaskResult:
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")

    if not task.image_id:
        return TaskResult(image_url=task.image_url)

    image_path = _load_image_path(task.image_id)
    if thumbnail_only:
        _, preview_url = _make_preview(image_path, 256)
        return TaskResult(image_id=task.image_id, image_url=preview_url)
    if preview_size:
        _, preview_url = _make_preview(image_path, preview_size)
        return TaskResult(image_id=task.image_id, image_url=preview_url)

    return TaskResult(image_id=task.image_id, image_url=task.image_url)


@app.post("/tasks/{task_id}/extract")
async def extract_task(task_id: str) -> dict[str, str]:
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return {"task_id": task_id, "status": task.status}


@app.get("/uploads/{image_id}")
async def download_image(image_id: str) -> FileResponse:
    path = _load_image_path(image_id)
    return FileResponse(path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
