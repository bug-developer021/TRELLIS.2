![](assets/teaser.webp)

# 原生且紧凑的结构化潜变量用于 3D 生成

<a href="https://arxiv.org/abs/2512.14692"><img src="https://img.shields.io/badge/Paper-Arxiv-b31b1b.svg" alt="Paper"></a>
<a href="https://huggingface.co/microsoft/TRELLIS.2-4B"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow" alt="Hugging Face"></a>
<a href="https://huggingface.co/spaces/microsoft/TRELLIS.2"><img src="https://img.shields.io/badge/Hugging%20Face-Demo-blueviolet" alt="Hugging Face"></a>
<a href="https://microsoft.github.io/TRELLIS.2"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Project Page"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>

https://github.com/user-attachments/assets/63b43a7e-acc7-4c81-a900-6da450527d8f

*(由于 GitHub 体积限制，视频为压缩版本。完整清晰版请见项目主页。)*

**TRELLIS.2** 是一款先进的 3D 大模型（40 亿参数），用于高保真 **图像到 3D** 生成。它采用一种全新的“无场”稀疏体素结构 **O-Voxel**，可重建并生成拓扑复杂、细节锐利且支持完整 PBR 材质的 3D 资产。


## ✨ 特性

### 1. 高质量、高分辨率与高效率
4B 参数模型能够生成高分辨率、带完整纹理的 3D 资产，同时保持优秀的效率，采用普通的 DiT 架构。模型使用 16× 空间下采样的稀疏 3D VAE 将资产编码到紧凑潜空间。

| 分辨率 | 总耗时* | 分解（形状 + 材质） |
| :--- | :--- | :--- |
| **512³** | **~3s** | 2s + 1s |
| **1024³** | **~17s** | 10s + 7s |
| **1536³** | **~60s** | 35s + 25s |

<small>*在 NVIDIA H100 GPU 上测试。</small>

### 2. 支持任意拓扑
**O-Voxel** 表示突破等值面场限制，可稳健处理复杂结构而无需有损转换：
*   ✅ **开放面**（例如衣物、叶片）
*   ✅ **非流形几何**
*   ✅ **内部封闭结构**

### 3. 丰富材质建模
除基础颜色外，TRELLIS.2 还能建模 **Base Color、Roughness、Metallic、Opacity** 等属性，支持写实渲染与透明材质。

### 4. 极简处理流程
数据处理非常高效，支持即时转换，**无需渲染、无需优化**：
*   **< 10s**（单 CPU）：纹理网格 → O-Voxel
*   **< 100ms**（CUDA）：O-Voxel → 纹理网格


## 🗺️ Roadmap

- [x] 论文发布
- [x] 发布图像到 3D 推理代码
- [x] 发布预训练检查点（4B）
- [x] Hugging Face Spaces Demo
- [ ] 发布形状条件纹理生成推理代码（计划：2025/12/24 前）
- [ ] 发布训练代码（计划：2025/12/31 前）


## 🛠️ 安装

### 先决条件
- **系统**：当前仅在 **Linux** 上测试。
- **硬件**：需要至少 24GB 显存的 NVIDIA GPU；已在 A100 和 H100 验证。
- **软件**：
  - 需要 [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) 编译部分依赖，推荐 12.4 版本。
  - 推荐使用 [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) 管理依赖。
  - 需要 Python 3.8 或更高版本。

### 安装步骤
1. 克隆仓库：
    ```sh
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
    cd TRELLIS.2
    ```

2. 安装依赖：

    **运行以下命令前请注意：**
    - 添加 `--new-env` 会创建新的 conda 环境 `trellis2`。若使用现有环境，请移除此参数。
    - 默认使用 CUDA 12.4 的 PyTorch 2.6.0。如需其他 CUDA 版本，请移除 `--new-env` 并手动安装依赖，参考 [PyTorch](https://pytorch.org/get-started/previous-versions/)。
    - 若系统存在多个 CUDA 版本，请在运行前设置 `CUDA_HOME` 指向正确版本，例如 `export CUDA_HOME=/usr/local/cuda-12.4`。
    - 默认使用 `flash-attn` 作为注意力后端。对于不支持 `flash-attn` 的 GPU（如 V100），可手动安装 `xformers` 并设置 `ATTN_BACKEND=xformers`。
    - 依赖较多，安装可能耗时，请耐心等待。
    - 如遇问题，可逐项安装依赖并分步排查。

    创建并安装依赖：
    ```sh
    . ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
    ```
    `setup.sh` 详见：
    ```sh
    Usage: setup.sh [OPTIONS]
    Options:
        -h, --help              Display this help message
        --new-env               Create a new conda environment
        --basic                 Install basic dependencies
        --flash-attn            Install flash-attention
        --cumesh                Install cumesh
        --o-voxel               Install o-voxel
        --flexgemm              Install flexgemm
        --nvdiffrast            Install nvdiffrast
        --nvdiffrec             Install nvdiffrec
    ```


## 📦 预训练权重

预训练模型 **TRELLIS.2-4B** 位于 Hugging Face，更多细节见模型卡。

| 模型 | 参数量 | 分辨率 | 链接 |
| :--- | :--- | :--- | :--- |
| **TRELLIS.2-4B** | 40 亿 | 512³ - 1536³ | [Hugging Face](https://huggingface.co/microsoft/TRELLIS.2-4B) |


## 🚀 使用

### 1. 图像到 3D 生成

#### 最小示例

以下是使用预训练模型生成 3D 资产的 [示例](example.py)：

```python
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 可节省显存
import cv2
import imageio
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

# 1. 配置环境贴图
envmap = EnvMap(torch.tensor(
    cv2.cvtColor(cv2.imread('assets/hdri/forest.exr', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
    dtype=torch.float32, device='cuda'
))

# 2. 加载 Pipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

# 3. 加载图像并运行
image = Image.open("assets/example_image/T.png")
mesh = pipeline.run(image)[0]
mesh.simplify(16777216) # nvdiffrast 限制

# 4. 渲染视频
video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
imageio.mimsave("sample.mp4", video, fps=15)

# 5. 导出 GLB
glb = o_voxel.postprocess.to_glb(
    vertices            =   mesh.vertices,
    faces               =   mesh.faces,
    attr_volume         =   mesh.attrs,
    coords              =   mesh.coords,
    attr_layout         =   mesh.layout,
    voxel_size          =   mesh.voxel_size,
    aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target   =   1000000,
    texture_size        =   4096,
    remesh              =   True,
    remesh_band         =   1,
    remesh_project      =   0,
    verbose             =   True
)
glb.export("sample.glb", extension_webp=True)
```

运行后生成：
 - `sample.mp4`：含 PBR 材质与环境光的渲染视频。
 - `sample.glb`：可直接用于渲染的 GLB 资产。

**注意**：`.glb` 默认以 `OPAQUE` 模式导出。虽然贴图包含 alpha，但默认未启用透明。若需透明，请在 3D 软件中将贴图 alpha 接入材质透明/alpha 输入。

#### Web Demo（前后端分离 + 异步后端）

Web Demo 分为 Gradio 前端（`app.py`）与异步 FastAPI 后端（`backend_service.py`）。
请先启动后端，再运行前端，并配置后端地址。

**启动后端服务**
```sh
export TRELLIS2_MAX_ACTIVE_TASKS=1     # GPU 并发任务上限
export TRELLIS2_WORKER_COUNT=2         # 异步队列 worker 数
uvicorn backend_service:app --host 0.0.0.0 --port 8000
```

**启动前端**
```sh
export TRELLIS2_BACKEND_URL=http://127.0.0.1:8000
python app.py
```

然后在终端输出的地址访问 Demo。

#### 后端 API 文档

所有接口均为 JSON 请求/响应（除非特别说明）。

**POST `/preprocess`** — 去背景与裁剪
```json
{
  "image_base64": "<base64 PNG bytes>"
}
```
响应：
```json
{
  "image_base64": "<base64 PNG bytes>"
}
```

**POST `/tasks`** — 提交异步生成任务
```json
{
  "image_base64": "<base64 PNG bytes>",
  "seed": 0,
  "resolution": "1024",
  "sampler_params": {
    "sparse_structure": {
      "steps": 12,
      "guidance_strength": 7.5,
      "guidance_rescale": 0.7,
      "rescale_t": 5.0
    },
    "shape_slat": {
      "steps": 12,
      "guidance_strength": 7.5,
      "guidance_rescale": 0.5,
      "rescale_t": 3.0
    },
    "tex_slat": {
      "steps": 12,
      "guidance_strength": 1.0,
      "guidance_rescale": 0.0,
      "rescale_t": 3.0
    }
  }
}
```
响应：
```json
{
  "task_id": "<uuid>"
}
```

**GET `/tasks/{task_id}`** — 查询任务状态
响应：
```json
{
  "status": "queued|running|succeeded|failed",
  "error": "optional error message"
}
```

**GET `/tasks/{task_id}/result`** — 获取渲染预览
响应：
```json
{
  "rendered": {
    "normal": ["data:image/jpeg;base64,...", "..."],
    "clay": ["data:image/jpeg;base64,...", "..."],
    "base_color": ["data:image/jpeg;base64,...", "..."],
    "shaded_forest": ["data:image/jpeg;base64,...", "..."],
    "shaded_sunset": ["data:image/jpeg;base64,...", "..."],
    "shaded_courtyard": ["data:image/jpeg;base64,...", "..."]
  },
  "resolution": "1024"
}
```

**POST `/tasks/{task_id}/extract`** — 导出 GLB
```json
{
  "decimation_target": 500000,
  "texture_size": 2048
}
```
响应：
```json
{
  "glb_path": "/absolute/path/to/sample_YYYY-MM-DDTHHMMSS.mmm.glb"
}
```

### 2. PBR 纹理生成

即将发布，敬请期待！

## 🧩 相关包

TRELLIS.2 基于多个高性能工具包：

*   **[O-Voxel](o-voxel)：**
    核心库，负责纹理网格与 O-Voxel 表示之间的双向转换。
*   **[FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM)：**
    基于 Triton 的高效稀疏卷积实现。
*   **[CuMesh](https://github.com/JeffreyXiang/CuMesh)：**
    CUDA 加速网格处理，包括高效后处理、重建、简化与 UV 展开。


## ⚖️ 许可

模型与代码采用 **[MIT License](LICENSE)**。

部分依赖有独立许可协议：

- [**nvdiffrast**](https://github.com/NVlabs/nvdiffrast): 用于渲染 3D 资产。
- [**nvdiffrec**](https://github.com/NVlabs/nvdiffrec): 用于 PBR 分裂求和渲染。

## 📚 引用

如对研究有帮助，请引用：

```bibtex
@article{
    xiang2025trellis2,
    title={Native and Compact Structured Latents for 3D Generation},
    author={Xiang, Jianfeng and Chen, Xiaoxue and Xu, Sicheng and Wang, Ruicheng and Lv, Zelong and Deng, Yu and Zhu, Hongyuan and Dong, Yue and Zhao, Hao and Yuan, Nicholas Jing and Yang, Jiaolong},
    journal={Tech report},
    year={2025}
}
```
