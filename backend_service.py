import base64
import io
import os
import uuid
import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils
from trellis2.modules.sparse import SparseTensor
import o_voxel


os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
MAX_ACTIVE_TASKS = int(os.environ.get("TRELLIS2_MAX_ACTIVE_TASKS", "1"))
WORKER_COUNT = int(os.environ.get("TRELLIS2_WORKER_COUNT", "2"))

MODES = [
    {"name": "Normal", "render_key": "normal"},
    {"name": "Clay render", "render_key": "clay"},
    {"name": "Base color", "render_key": "base_color"},
    {"name": "HDRI forest", "render_key": "shaded_forest"},
    {"name": "HDRI sunset", "render_key": "shaded_sunset"},
    {"name": "HDRI courtyard", "render_key": "shaded_courtyard"},
]
STEPS = 8

app = FastAPI(title="TRELLIS.2 Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[Trellis2ImageTo3DPipeline] = None
envmap = None

task_queue: asyncio.Queue[str] = asyncio.Queue()
semaphore = asyncio.Semaphore(MAX_ACTIVE_TASKS)


class SamplerParams(BaseModel):
    steps: int = 12
    guidance_strength: float = 7.5
    guidance_rescale: float = 0.7
    rescale_t: float = 5.0


class TaskSamplerParams(BaseModel):
    sparse_structure: SamplerParams = Field(default_factory=SamplerParams)
    shape_slat: SamplerParams = Field(default_factory=lambda: SamplerParams(guidance_rescale=0.5, rescale_t=3.0))
    tex_slat: SamplerParams = Field(default_factory=lambda: SamplerParams(guidance_strength=1.0, guidance_rescale=0.0, rescale_t=3.0))


class TaskRequest(BaseModel):
    image_base64: str
    seed: int = 0
    resolution: str = "1024"
    sampler_params: TaskSamplerParams = Field(default_factory=TaskSamplerParams)


class PreprocessRequest(BaseModel):
    image_base64: str


class ExtractRequest(BaseModel):
    decimation_target: int = 500000
    texture_size: int = 2048


@dataclass
class TaskRecord:
    task_id: str
    status: str = "queued"
    error: Optional[str] = None
    result: Optional[Dict] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


tasks: Dict[str, TaskRecord] = {}


def init_models() -> None:
    global pipeline, envmap
    if pipeline is not None:
        return
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
    pipeline.cuda()
    envmap = {
        'forest': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
        'sunset': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/sunset.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
        'courtyard': EnvMap(torch.tensor(
            cv2.cvtColor(cv2.imread('assets/hdri/courtyard.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
            dtype=torch.float32, device='cuda'
        )),
    }


def decode_image(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGBA")


def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image = image.convert("RGB")
    image.save(buffered, format="jpeg", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"


def pack_state(latents: tuple, path: str) -> None:
    shape_slat, tex_slat, res = latents
    payload = {
        'shape_slat_feats': shape_slat.feats.cpu().numpy(),
        'tex_slat_feats': tex_slat.feats.cpu().numpy(),
        'coords': shape_slat.coords.cpu().numpy(),
        'res': res,
    }
    np.savez(path, **payload)


def unpack_state(path: str) -> tuple:
    data = np.load(path)
    shape_slat = SparseTensor(
        feats=torch.from_numpy(data['shape_slat_feats']).cuda(),
        coords=torch.from_numpy(data['coords']).cuda(),
    )
    tex_slat = shape_slat.replace(torch.from_numpy(data['tex_slat_feats']).cuda())
    return shape_slat, tex_slat, int(data['res'])


def render_preview(mesh) -> Dict[str, List[str]]:
    images = render_utils.render_snapshot(mesh, resolution=1024, r=2, fov=36, nviews=STEPS, envmap=envmap)
    rendered = {}
    for mode in MODES:
        rendered[mode["render_key"]] = [
            image_to_base64(Image.fromarray(img)) for img in images[mode["render_key"]]
        ]
    return rendered


def run_inference(task_id: str, request: TaskRequest) -> Dict:
    init_models()
    image = decode_image(request.image_base64)
    image = pipeline.preprocess_image(image)
    outputs, latents = pipeline.run(
        image,
        seed=request.seed,
        preprocess_image=False,
        sparse_structure_sampler_params=request.sampler_params.sparse_structure.model_dump(),
        shape_slat_sampler_params=request.sampler_params.shape_slat.model_dump(),
        tex_slat_sampler_params=request.sampler_params.tex_slat.model_dump(),
        pipeline_type={
            "512": "512",
            "1024": "1024_cascade",
            "1536": "1536_cascade",
        }[request.resolution],
        return_latent=True,
    )
    mesh = outputs[0]
    mesh.simplify(16777216)
    rendered = render_preview(mesh)
    torch.cuda.empty_cache()

    task_dir = os.path.join(TMP_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    state_path = os.path.join(task_dir, "state.npz")
    pack_state(latents, state_path)

    return {
        "rendered": rendered,
        "state_path": state_path,
        "resolution": request.resolution,
    }


async def worker() -> None:
    while True:
        task_id = await task_queue.get()
        record = tasks[task_id]
        record.status = "running"
        async with semaphore:
            try:
                request = record.result["request"]
                result = await asyncio.to_thread(run_inference, task_id, request)
                record.result = result
                record.status = "succeeded"
            except Exception as exc:
                record.status = "failed"
                record.error = str(exc)
        task_queue.task_done()


@app.on_event("startup")
async def startup() -> None:
    os.makedirs(TMP_DIR, exist_ok=True)
    for _ in range(WORKER_COUNT):
        asyncio.create_task(worker())


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/preprocess")
def preprocess_image(payload: PreprocessRequest) -> Dict[str, str]:
    init_models()
    image = decode_image(payload.image_base64)
    processed = pipeline.preprocess_image(image)
    buffered = io.BytesIO()
    processed.save(buffered, format="PNG")
    return {"image_base64": base64.b64encode(buffered.getvalue()).decode()}


@app.post("/tasks")
def submit_task(payload: TaskRequest) -> Dict[str, str]:
    task_id = str(uuid.uuid4())
    tasks[task_id] = TaskRecord(task_id=task_id, status="queued", result={"request": payload})
    task_queue.put_nowait(task_id)
    return {"task_id": task_id}


@app.get("/tasks/{task_id}")
def get_task_status(task_id: str) -> Dict[str, str]:
    record = tasks.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Task not found")
    response = {"status": record.status}
    if record.error:
        response["error"] = record.error
    return response


@app.get("/tasks/{task_id}/result")
def get_task_result(task_id: str) -> Dict:
    record = tasks.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if record.status != "succeeded":
        raise HTTPException(status_code=400, detail="Task not completed")
    return {
        "rendered": record.result["rendered"],
        "resolution": record.result["resolution"],
    }


@app.post("/tasks/{task_id}/extract")
def extract_glb(task_id: str, payload: ExtractRequest) -> Dict[str, str]:
    record = tasks.get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if record.status != "succeeded":
        raise HTTPException(status_code=400, detail="Task not completed")

    state_path = record.result["state_path"]
    if not os.path.exists(state_path):
        raise HTTPException(status_code=404, detail="Task state not found")

    shape_slat, tex_slat, res = unpack_state(state_path)
    mesh = pipeline.decode_latent(shape_slat, tex_slat, res)[0]
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=res,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=payload.decimation_target,
        texture_size=payload.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        use_tqdm=True,
    )

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H%M%S") + f".{now.microsecond // 1000:03d}"
    task_dir = os.path.join(TMP_DIR, task_id)
    os.makedirs(task_dir, exist_ok=True)
    glb_path = os.path.join(task_dir, f"sample_{timestamp}.glb")
    glb.export(glb_path, extension_webp=True)
    torch.cuda.empty_cache()
    return {"glb_path": glb_path}
