import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel

from storage.task_store import TaskRecord, TaskStore


TMP_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = os.getenv("TRELLIS2_TASK_DB", str(TMP_DIR / "tasks.db"))
TASK_TTL_SECONDS = int(os.getenv("TASK_TTL_SECONDS", "86400"))
CLEANUP_INTERVAL_SECONDS = int(os.getenv("TASK_CLEANUP_INTERVAL_SECONDS", "300"))
MAX_ACTIVE_TASKS = int(os.getenv("TRELLIS2_MAX_ACTIVE_TASKS", "2"))

DEFAULT_PREVIEW_SIZE = int(os.getenv("TRELLIS2_PREVIEW_SIZE", "512"))
THUMBNAIL_PREVIEW_SIZE = int(os.getenv("TRELLIS2_THUMBNAIL_PREVIEW_SIZE", "256"))

API_KEY = os.getenv("TRELLIS2_API_KEY")
BEARER_TOKEN = os.getenv("TRELLIS2_BEARER_TOKEN")

store = TaskStore(DB_PATH)

app = FastAPI()

queue_lock = threading.Lock()
queue: Deque[str] = deque()
running_tasks: set[str] = set()


class TaskCreateRequest(BaseModel):
    params: Dict[str, Any] = {}
    preview_mode: Optional[str] = None


class TaskResponse(BaseModel):
    task_id: str
    status: str
    created_at: float
    params: Dict[str, Any]
    result_path: Optional[str]
    preview_size: int
    preview_mode: str
    preview_images: List[str]
    canceled: bool


def get_owner(request: Request) -> str:
    auth_header = request.headers.get("Authorization")
    api_key = request.headers.get("X-API-Key")

    if API_KEY or BEARER_TOKEN:
        if api_key and API_KEY and api_key == API_KEY:
            return f"api-key:{api_key}"
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]
            if BEARER_TOKEN and token == BEARER_TOKEN:
                return f"bearer:{token}"
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    return "anonymous"


def task_dir(task_id: str) -> Path:
    return TMP_DIR / task_id


def list_preview_images(task_id: str, preview_mode: str) -> List[str]:
    previews = sorted(task_dir(task_id).glob("preview_*.png"))
    if preview_mode == "thumbnail":
        previews = previews[:2]
    return [str(path) for path in previews]


def cleanup_task_files(task_id: str) -> None:
    task_path = task_dir(task_id)
    if not task_path.exists():
        return
    state_path = task_path / "state.npz"
    if state_path.exists():
        state_path.unlink()
    for glb_file in task_path.glob("sample_*.glb"):
        glb_file.unlink()
    for preview_file in task_path.glob("preview_*.png"):
        preview_file.unlink()
    try:
        task_path.rmdir()
    except OSError:
        pass


def cleanup_loop() -> None:
    while True:
        time.sleep(CLEANUP_INTERVAL_SECONDS)
        for task in list(store.list_expired(TASK_TTL_SECONDS)):
            cleanup_task_files(task.task_id)
            store.delete_task(task.task_id)


def start_cleanup_thread() -> None:
    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()


@app.on_event("startup")
async def startup_event() -> None:
    start_cleanup_thread()


@app.post("/tasks", response_model=TaskResponse)
async def create_task(payload: TaskCreateRequest, owner: str = Depends(get_owner)) -> TaskResponse:
    if store.count_active_by_owner(owner) >= MAX_ACTIVE_TASKS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many active tasks for this user",
        )

    task_id = uuid.uuid4().hex
    preview_mode = payload.preview_mode or "full"
    preview_size = (
        THUMBNAIL_PREVIEW_SIZE if preview_mode == "thumbnail" else DEFAULT_PREVIEW_SIZE
    )

    store.create_task(
        task_id=task_id,
        status="queued",
        params=payload.params,
        owner=owner,
        preview_size=preview_size,
        preview_mode=preview_mode,
    )

    with queue_lock:
        queue.append(task_id)

    return TaskResponse(
        task_id=task_id,
        status="queued",
        created_at=time.time(),
        params=payload.params,
        result_path=None,
        preview_size=preview_size,
        preview_mode=preview_mode,
        preview_images=[],
        canceled=False,
    )


@app.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str, owner: str = Depends(get_owner)) -> TaskResponse:
    task = store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    if task.owner != owner and owner != "anonymous":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    preview_images = list_preview_images(task.task_id, task.preview_mode)
    return TaskResponse(
        task_id=task.task_id,
        status=task.status,
        created_at=task.created_at,
        params=task.params,
        result_path=task.result_path,
        preview_size=task.preview_size,
        preview_mode=task.preview_mode,
        preview_images=preview_images,
        canceled=task.canceled,
    )


@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str, owner: str = Depends(get_owner)) -> Dict[str, Any]:
    task = store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    if task.owner != owner and owner != "anonymous":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    with queue_lock:
        if task_id in queue:
            queue.remove(task_id)
            store.mark_canceled(task_id)
            return {"task_id": task_id, "status": "canceled"}

    store.mark_canceled(task_id)
    running_tasks.discard(task_id)
    return {"task_id": task_id, "status": "canceled"}


@app.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str, owner: str = Depends(get_owner)) -> Dict[str, Any]:
    task = store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found")
    if task.owner != owner and owner != "anonymous":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return {"task_id": task_id, "result_path": task.result_path}
