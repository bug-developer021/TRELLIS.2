# TRELLIS.2 图像生成 3D 资产流程文档

> 基于仓库源码阅读整理，面向图像到 3D 资产的端到端推理流程说明。

## 1. 总览流程（含 Mermaid 流程图）

```mermaid
flowchart TD
    A[输入图像 Image (PIL)] --> B[预处理: preprocess_image]
    B -->|去背景/裁剪/缩放| C[条件特征提取: image_cond_model]
    C -->|cond_512/cond_1024| D[稀疏结构采样: sparse_structure_flow + sampler]
    D -->|稀疏体素坐标 coords| E[形状 SLat 采样: shape_slat_flow + sampler]
    E -->|shape_slat| F[形状解码: shape_slat_decoder]
    F -->|Mesh + subs| G[纹理 SLat 采样: tex_slat_flow + sampler]
    G -->|tex_slat| H[纹理解码: tex_slat_decoder]
    H -->|纹理体素 attrs| I[MeshWithVoxel 输出]
    I -->|可选导出| J[o_voxel.postprocess.to_glb → GLB]
```

### 模块清单与输入输出

| 模块 | 作用 | 输入 | 输出 |
| --- | --- | --- | --- |
| `preprocess_image` | 统一图像尺度、前景提取、裁剪 | PIL Image | PIL Image |
| `image_cond_model` | 提取图像条件特征 | Image/Batch | Patch Tokens |
| `sparse_structure_flow` | 生成稀疏结构 latent | 噪声+条件特征 | 稀疏结构 latent |
| `sparse_structure_decoder` | 解码稀疏 occupancy | latent | 稀疏坐标 coords |
| `shape_slat_flow` | 生成形状结构化潜变量 | coords + 条件 | shape_slat |
| `shape_slat_decoder` | 解码形状网格 | shape_slat | Mesh + subs |
| `tex_slat_flow` | 生成纹理结构化潜变量 | shape_slat + 条件 | tex_slat |
| `tex_slat_decoder` | 解码纹理体素 | tex_slat + subs | attrs |
| `MeshWithVoxel` | 组装网格与纹理体素 | Mesh + attrs | 3D 资产 |
| `to_glb` | PBR 导出 | MeshWithVoxel | GLB |

## 2. 详细模块说明（原理 + 优势对比）

### 2.1 图像预处理（`preprocess_image` + `BiRefNet`）

**位置**：`trellis2/pipelines/trellis2_image_to_3d.py`，`trellis2/pipelines/rembg/BiRefNet.py`

**原理**：
- 若输入图像无有效 alpha，则使用 BiRefNet 做前景抠图生成 alpha。
- 使用 alpha 计算前景 bbox，裁剪并缩放到 <= 1024。
- 最终输出前景颜色（RGB * alpha）以降低背景干扰。

**优势**：
- 提升形状与纹理建模的信噪比。
- 统一尺度，减少后续模型对不同图像尺寸的偏差。

**对比**：
- **无抠图**：背景噪声更高，结构偏差大。
- **自动抠图**：前景更干净，对几何与材质生成更稳定。

---

### 2.2 图像条件特征提取（`DinoV2FeatureExtractor` / `DinoV3FeatureExtractor`）

**位置**：`trellis2/modules/image_feature_extractor.py`

**原理**：
- 将输入图像编码为 patch-level 语义特征（Transformer backbone）。
- 特征作为跨注意力条件引导形状与纹理生成。

**优势**：
- DINO 系列对语义结构、材质纹理均有较强表征能力。
- Patch tokens 可兼容稀疏/密集混合注意力。

**对比**：
| 方案 | 优势 | 局限 |
| --- | --- | --- |
| DINOv2 | 语义稳定、对纹理鲁棒 | 预训练固定，领域迁移有限 |
| DINOv3 | 表征更强，适合高细节 | 计算量更大 |

---

### 2.3 稀疏结构采样（`SparseStructureFlowModel` + Decoder）

**位置**：`trellis2/models/sparse_structure_flow.py`，`trellis2/models/sparse_structure_vae.py`

**原理**：
- 使用 flow 模型在稠密体素空间采样结构 latent。
- 通过稀疏结构 decoder 生成 occupancy 体素，并提取稀疏坐标。

**优势**：
- 显著减少后续 shape/texture 的 token 数。
- 先确定“结构骨架”，降低高分辨率生成难度。

**对比**：
| 方案 | 优势 | 局限 |
| --- | --- | --- |
| 稀疏结构 | 计算高效，支持复杂拓扑 | 需要额外结构预测步骤 |
| Dense voxel | 实现简单 | 计算量巨大 |

---

### 2.4 形状 SLat 采样（`SLatFlowModel`）

**位置**：`trellis2/models/structured_latent_flow.py`

**原理**：
- 在稀疏结构坐标上进行结构化潜变量采样。
- 支持级联（512 → 1024/1536）逐步提升分辨率。

**优势**：
- 稀疏 token 推理更快。
- 级联采样更稳定、细节更丰富。

**对比**：
| 方案 | 优势 | 局限 |
| --- | --- | --- |
| 级联 SLat | 稳定、细节递进 | 多阶段推理耗时更长 |
| 单阶段高分辨率 | 简单 | 更容易失败、显存占用高 |

---

### 2.5 形状解码（`FlexiDualGridVaeDecoder`）

**位置**：`trellis2/models/sc_vaes/fdg_vae.py`

**原理**：
- 将 shape_slat 解码为网格（Mesh），并输出 subs 供纹理解码。
- 使用 flexible dual grid 方式生成高质量网格。

**优势**：
- 可支持复杂拓扑（开放面、非流形等）。
- 网格质量高，适合 PBR 贴图。

---

### 2.6 纹理 SLat 采样与解码

**位置**：`trellis2/pipelines/trellis2_image_to_3d.py` (`sample_tex_slat`, `decode_tex_slat`)

**原理**：
- 将 shape_slat 与噪声拼接后再采样纹理潜变量。
- 纹理解码输出 PBR 属性：Base Color / Metallic / Roughness / Alpha。

**优势**：
- 纹理与几何对齐（形状潜变量为强条件）。
- 直接输出 PBR 属性，适合渲染与导出。

**对比**：
| 方案 | 优势 | 局限 |
| --- | --- | --- |
| 条件纹理生成 | 几何一致性强 | 依赖形状质量 |
| 无条件纹理生成 | 独立性强 | 容易错位 |

---

## 3. 模块调用关系（函数级别）

入口：`Trellis2ImageTo3DPipeline.run(...)`

调用顺序：
1. `preprocess_image`：前景提取 + 裁剪缩放
2. `get_cond`：提取 512/1024 图像条件特征
3. `sample_sparse_structure`：采样稀疏结构 coords
4. `sample_shape_slat` 或 `sample_shape_slat_cascade`
5. `sample_tex_slat`
6. `decode_latent`
   - `decode_shape_slat` → Mesh + subs
   - `decode_tex_slat` → attrs
7. 输出 `MeshWithVoxel`

## 4. 模型微调可行性分析

> 说明：训练代码尚未发布（README Roadmap），以下为基于模块结构与常见实践的可行性分析。

### 4.1 可微调模块与预期效果

| 模块 | 可微调方式 | 预期效果 |
| --- | --- | --- |
| `image_cond_model` (DINO) | LoRA / Adapter | 提升领域特定语义对齐 |
| `sparse_structure_flow` | LoRA / 全参 | 稀疏结构更准确、更稳定 |
| `shape_slat_flow` | LoRA / 全参 | 几何细节更丰富 |
| `shape_slat_decoder` | 全参 | 网格质量提升，孔洞更少 |
| `tex_slat_flow` | LoRA / 全参 | 材质一致性和风格控制 |
| `tex_slat_decoder` | 全参 | PBR 贴图细节增强 |
| `BiRefNet` | 替换/微调 | 更准确的前景分割 |

### 4.2 微调数据与标注需求

**形状相关（结构/几何）**
- **输入**：图像
- **监督**：高质量网格（可转 O-Voxel 或结构化 latent）
- **标注**：几何质量、噪声剔除、尺度归一

**纹理相关（材质）**
- **输入**：图像
- **监督**：PBR 贴图（BaseColor / Metallic / Roughness / Alpha）
- **标注**：UV 展开、材质分离、纹理一致性

**端到端微调（整体）**
- **输入**：图像
- **监督**：几何 + PBR 贴图 + 一致尺度
- **需求**：完整训练 pipeline 与一致的数据处理链

### 4.3 计算资源与时间估计（经验量级）

| 微调策略 | GPU 资源 | 估计时间 |
| --- | --- | --- |
| LoRA / Adapter (DINO/Flow) | 1×A100/H100 | 数小时 ~ 1 天 |
| 子模块全参 | 4~8×A100/H100 | 2~7 天 |
| 端到端大规模 | 8×A100/H100 | 1~3 周 |

## 5. 模块间调用关系总结

- **图像预处理** → **条件特征提取** → **稀疏结构采样** → **形状 SLat 采样与解码** → **纹理 SLat 采样与解码** → **MeshWithVoxel 输出** → **GLB 导出**。
- 纹理分支依赖形状潜变量，确保纹理与几何一致。
- 级联模式通过先低分辨率，再高分辨率的方式提高稳定性。

---

## 6. 参考源码路径（便于定位）

- `trellis2/pipelines/trellis2_image_to_3d.py`
- `trellis2/modules/image_feature_extractor.py`
- `trellis2/models/sparse_structure_flow.py`
- `trellis2/models/sparse_structure_vae.py`
- `trellis2/models/structured_latent_flow.py`
- `trellis2/models/sc_vaes/fdg_vae.py`
- `trellis2/representations/mesh/base.py`
- `trellis2/pipelines/rembg/BiRefNet.py`
- `o_voxel/postprocess.py`
