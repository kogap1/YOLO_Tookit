# PTQ 全链路优化与验证：YOLO26 量化部署（RK3588）

---

## 一、课程概述

**核心目标**

- 理解 PTQ 量化原理与误差来源
- 掌握 YOLO26 ONNX → RKNN 量化全流程
- 能处理 YOLO26 INT8 量化的已知问题并选择最优策略
- 能用 RKNN C++ API 完成板端推理与性能测试
- 能输出完整的量化优化报告

**建议时长**

7–9 小时（含动手实验与故障排查）

**前置知识**

Python 基础、深度学习基础（CNN / BN / 激活函数）、了解 YOLO 检测头结构、C++ 基本语法

**验收标准**

- 量化后 mAP@0.5 下降 < 1%
- INT8 推理速度较 FP32 提升 ≥ 40%
- 能独立完成 RKNN C++ 推理代码编译运行
- 能生成含性能/精度对比表的优化报告

**硬件要求**

- 目标设备：RK3588 开发板（含 NPU 驱动）
- 导出主机：x86 Ubuntu（RKNN Toolkit2 ≥ v2.3.0）

---

## 二、知识点讲解

### 2.1 PTQ 量化基础原理

#### 2.1.1 什么是 PTQ

Post-Training Quantization（训练后量化）是指在模型训练完成后，不修改权重，直接对模型进行量化。相比于 QAT（Quantization-Aware Training），PTQ 无需重新训练，部署成本低，是端侧部署的首选方案。

#### 2.1.2 量化映射公式

INT8 非对称量化（RKNN 默认 `w8a8`）的通用映射：

$$
x_{int} = \text{round}\left(\frac{x_{float}}{s}\right) + z
$$

$$
x_{float} \approx (x_{int} - z) \times s
$$

| 符号 | 含义 | 确定方式 |
|------|------|----------|
| $x_{float}$ | 浮点值 | 原始数据 |
| $x_{int}$ | 量化后整数值（[0, 255]） | 计算结果 |
| $s$ (scale) | 量化步长 | $\frac{\max(x_{float}) - \min(x_{float})}{255}$ |
| $z$ (zero point) | 零点偏移 | $\text{round}\left(-\frac{\min(x_{float})}{s}\right)$ |

#### 2.1.3 量化粒度对比

| 粒度 | 参数组数 | 精度 | 硬件执行路径 | 计算开销 |
|------|----------|------|-------------|----------|
| **Per-Tensor** | 1 组 ($s$, $z$) 覆盖整个张量 | 低 | W8A8：一次 scale 乘加 | 最小 |
| **Per-Channel**（RKNN 默认） | 每个输出通道独立 1 组 | 高 | W8A8：per-channel scale **融合进 INT8 Conv 微码**，NPU 原生路径 | 极小 |
| **Per-Group** (32/64/128/256) | 每 N 个通道共享 1 组 | 最高（w4a16 场景） | W4A16：需**解包 4-bit 权重 → per-group 反量化 → 矩阵乘**，多一步 dequant 硬件操作 | 较大（w4a16 专有开销） |

> **为什么 Per-Channel 参数组数比 Per-Group(32) 多，开销却更小？**
>
> 开销大小不是由参数组数决定的，而是由 **bit-width + 硬件执行路径** 决定：
> - Per-Channel 跑在 **W8A8 原生硬件路径**上——per-channel scale 乘法已被编码进 NPU 的 INT8 卷积微码，一条指令完成
> - Per-Group 跑在 **W4A16 路径**上——需要额外硬件步骤：从内存 unpack 4-bit 权重 → 按 group 反量化到 FP16 → 再送入矩阵乘法器
>
> **为什么 Per-Group(W4A16) 精度列标注"最高"？**
>
> 精度不只看权重量化位宽，而看 **权重误差 + 激活误差** 的总和：
> - W8A8 Per-Channel：权重 8-bit（精确）+ 激活 8-bit（粗糙）→ 瓶颈在激活量化
> - W4A16 Per-Group：权重 4-bit（粗糙）+ 激活 FP16（无损）→ 瓶颈在权重被 group 补偿
>
> 激活值的 8-bit 量化误差通常 **远大于** 权重的 4-bit 量化误差，因为激活值分布受输入影响剧烈、范围不稳定。W4A16 把"省下的位宽"分配给激活值（不做量化），对激活敏感的模型（多数视觉模型）总精度反而更高。
>
> **工程结论**：`quantized_method='channel'`（W8A8 Per-Channel）是 RK3588 NPU **W8A8 下的最优原生路径**，精度与效率平衡最佳。

#### 2.1.4 量化误差三大来源与修复方向

| 误差来源 | 原因 | 修复方向 |
|----------|------|----------|
| **权重量化误差** | 权重分布范围广，均匀量化丢失尾部信息 | 切换 `mmse` 算法、混合量化对敏感层保持 FP16 |
| **激活值量化误差** | 激活值分布因输入不同而变化，校准集不够代表 | 增加校准集多样性、使用 KL 散度校准 |
| **算子融合误差** | BN 融合、Conv+ReLU 融合引入的数值累积偏差 | 降低 `optimization_level`、检查算子融合正确性 |

### 2.2 YOLO26 架构核心特性

#### 2.2.1 概述

YOLO26 是 Ultralytics 2026 年 1 月发布的新一代 YOLO 模型（`ultralytics ≥ 8.4.0`），最核心的变革是 **NMS-Free 端到端架构**。

#### 2.2.2 双头架构（Dual-Head）

| 头 (Head) | 用途 | 输出形状 (640×640 COCO 80类) | 后处理 |
|-----------|------|------------------------------|--------|
| **One-to-One** (端到端) | 推理/导出默认 | `(1, 300, 6)` | **仅置信度过滤** |
| **One-to-Many** (传统) | 训练辅助 + `end2end=False` 回退 | `(1, 84, 8400)` | NMS |

- `model.fuse()` 将 O2M 头彻底移除，减小模型体积
- 导出默认使用 O2O 头

#### 2.2.3 输出格式详解

**端到端模式（默认）**：
```
形状: (batch, 300, 6)
格式: [x1, y1, x2, y2, confidence, class_id]
坐标: xyxy (左上+右下)，归一化到 [0, 1]
300:  最多检测数（可调 max_det）
```

**传统模式 (`end2end=False`)**：
```
形状: (batch, 84, 8400)
格式: 84 = 4(bbox) + 80(cls)  — 无 objectness!
需要: NMS 后处理
```

#### 2.2.4 YOLO26 与其他 YOLO 对比

| 特性 | YOLOv8 | YOLOv6 (美团) | YOLO11 | YOLO26 |
|------|--------|-------------|--------|--------|
| **DFL** | 有 | 无 | 有 | **无** |
| **Objectness** | 无 | 有 | 无 | **无** |
| **NMS** | 需要 | 需要 | 需要 | **不需要** |
| **输出格式（默认）** | 3 个检测头 | (1, 8400, 85) | 3 个检测头 | **(1, 300, 6)** |
| **坐标格式** | xywh (中心) | cx,cy,w,h | xywh (中心) | **xyxy (角点)** |
| **后处理复杂度** | DFL+NMS | conf+NMS | DFL+NMS | **仅 conf 过滤** |
| **C++ 后处理代码量** | ~150 行 | ~80 行 | ~150 行 | **~15 行** |

#### 2.2.5 关键架构创新

| 创新 | 说明 | 对量化的影响 |
|------|------|------------|
| **移除 DFL** | 回归头直接输出 bbox，不需要 Distribution Focal Loss 的 softmax 积分 | 消除了一大 INT8 量化误差源 |
| **NMS-Free** | 模型内部通过 O2O 匹配完成去重 | 消除了 NMS 参数调优和后处理延迟 |
| **MuSGD 优化器** | SGD + Muon 混合，收敛更稳定 | 训练出的权重分布更规整，有利于量化 |
| **ProgLoss + STAL** | 渐进式损失平衡 + 小目标感知标签分配 | 小目标精度提升，量化退化监控更有意义 |

### 2.3 YOLO26 INT8 量化的已知挑战

> ⚠️ **重要**：YOLO26 + RK3588 的 INT8 量化目前处于早期阶段，存在已知问题。

#### 2.3.1 当前状态矩阵

| 量化模式 | 状态 | 说明 |
|----------|------|------|
| **FP16 RKNN** | ✅ 稳定 | 检测正确，精度好，推荐生产使用 |
| **INT8 (E2E=True)** | ❌ 已知问题 | 可能触发 Segmentation Fault 或检测完全崩溃 |
| **INT8 (E2E=False)** | ⚠️ 部分可用 | 原始输出正常，但需 CPU 端 NMS |
| **混合量化 (auto_hybrid)** | ⚠️ 实验性 | 保护输出头会触发 `Op [exDataConvert] not support` 错误 |

#### 2.3.2 问题根源分析

```
YOLO26 INT8 量化坍塌的三大原因:

1. O2O 输出头权重稀疏
   → INT8 量化后 → 目标置信度全部归零 → 300 个输出全是 [0,0,0,0,0,-1]

2. 一对一头缺少 O2M 的冗余性
   → 每个位置只预测一个框 → 量化误差无容错空间

3. 输出头最后一层无 Sigmoid（O2O 内部做）
   → RKNN 优化器激进融合 → Sigmoid 被融入 INT8 Conv → 精度崩溃
```

> 来源：GitHub Issue [#23753](https://github.com/ultralytics/ultralytics/issues/23753) (ultralytics/ultralytics) 社区实测验证

#### 2.3.3 应对策略

| 策略 | 做法 | 精度 | 速度 | 推荐度 |
|------|------|------|------|--------|
| **策略A: FP16（纯）** | 全图 FP16 RKNN | 无损（实测 mAP 下降 < 0.2%） | 实测 76.97ms（13 FPS） | ⭐⭐⭐⭐⭐ 生产首选 |
| **策略B: E2E=False + INT8** | 回退到传统 O2M 输出，CPU 端 NMS | 量化后精度可控 | 实测 42.56ms（23.5 FPS） | ⭐⭐⭐⭐ 参考方案 |
| **策略B+: 6头分离 + INT8** | ONNX 图拆分为 reg/cls 独立输出，各自量化 | 实测 mAP@0.5 = 0.669 | **实测 38.39ms（26.0 FPS）** | ⭐⭐⭐⭐⭐ 本项目实际方案 |
| **策略C: E2E=True + INT8 + 混合量化** | 手动保护 O2O 头各层为 FP16 | 待验证 | 待验证 | ⭐⭐ 实验性 |

> **本项目实际采用策略B+（6头分离 + INT8）**：在策略B基础上，将 ONNX 输出拆分为 6 个独立节点（3 尺度 × reg/cls），让 RKNN 为每个头分配独立量化参数，避免回归和分类争抢同一 scale。实测 INT8 量化稳定不崩溃，mAP@0.5 = 0.669（降幅 2.27%），推理 38.39ms（26 FPS），较 FP16 提速 50%。精度未达 1% 目标需进一步调优（详见第五章）。

### 2.4 ONNX 模型导出规范

#### 2.4.1 基本导出

```python
from ultralytics import YOLO

model = YOLO("yolo26s.pt")

# 策略B: 传统模式导出（兼容 INT8 量化）
model.export(
    format="onnx",
    opset=12,
    imgsz=(640, 640),
    simplify=True,
    dynamic=False,
    batch=1,
    end2end=False          # ← 关键：关闭端到端，输出传统格式
)
# 输出: yolo26s.onnx，输出形状 (1, 84, 8400)
```

#### 2.4.2 策略B+：6 头分离 ONNX 修改（本项目实际方案）

在标准导出后，对 ONNX 计算图做手术——在检测头 3 个 Conv 输出后插入 Split 节点，将 84 通道拆为 reg(4ch) + cls(80ch)，生成 6 个独立输出：

```
原始图:  Conv0+Conv1+Conv2 → Concat → Reshape → [output: 1×84×8400]

修改后:  Conv0 → Split(4+80) → output0_reg (1×4×6400)
                            → output0_cls (1×80×6400)
         Conv1 → Split(4+80) → output1_reg (1×4×1600)
                            → output1_cls (1×80×1600)
         Conv2 → Split(4+80) → output2_reg (1×4×400)
                            → output2_cls (1×80×400)
```

> **为什么要拆分？** 回归输出（坐标偏移，范围小）和分类输出（logits，范围大）数值分布差异巨大。合并时 RKNN 只分配一套 `(s,z)`，顾此失彼。拆分为 6 个独立输出后，各头获得独立量化参数，精度显著改善。

#### 2.4.3 导出参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `end2end` | **False**（当前策略） | True → (1,300,6) NMS-Free；False → (1,84,8400) 传统 |
| `opset` | **12** | RKNN Toolkit2 适配最成熟的版本 |
| `simplify` | **True** | onnxslim 精简图结构 |
| `dynamic` | **False** | 固定尺寸，NPU 编译最优 |
| `batch` | **1** | 单帧推理 |

> ⚠️ **验证步骤**（6 头方案）：用 Netron 检查：(1) Input: `[1,3,640,640]`；(2) 6 个 Output: `output0_reg [1,4,6400]`, `output0_cls [1,80,6400]`, `output1_reg [1,4,1600]`, `output1_cls [1,80,1600]`, `output2_reg [1,4,400]`, `output2_cls [1,80,400]`；(3) 图中无 NMS/Sigmoid/Softmax 等后处理节点。

### 2.5 RKNN Toolkit2 量化配置全参数

#### 2.5.1 配置模板

```python
from rknn.api import RKNN
rknn = RKNN()

# 本项目实际配置（默认参数，见 export_rknn_yolov26.py）
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    # 以下参数未显式设置，使用 RKNN 默认值：
    # quantized_dtype='w8a8'            默认
    # quantized_algorithm='normal'      默认（MinMax）
    # quantized_method='channel'        默认（逐通道）
    # optimization_level=3              默认（最高融合）
    target_platform='rk3588'
)
# 通过 --int8 开关控制: do_quantization=True/False
rknn.build(do_quantization=opt.int8, dataset=opt.data_path)
```

**当前配置说明**：
- `quantized_algorithm='normal'`（MinMax）→ YOLO26 激活值长尾分布，建议改用 `kl_divergence`
- `optimization_level=3` → 对检测头融合过于激进，建议降至 2
- 校准集仅 300 张 → 远低于推荐的 100-300 张

**改进后的推荐配置**：

```python
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    quantized_dtype='w8a8',
    quantized_algorithm='kl_divergence',   # 推荐：KL 散度
    quantized_method='channel',
    optimization_level=2,                  # 推荐：降低融合等级
    target_platform='rk3588'
)
```

#### 2.5.2 量化算法对比

| 算法 | 原理 | 速度 | 精度 | 推荐校准集 | 适用场景 |
|------|------|------|------|-----------|----------|
| `normal` (默认) | Min-Max 线性映射 | 快（秒级） | 中等 | 20–100 张 | 快速基线 |
| `kl_divergence` | KL 散度优化阈值 | 中等（分钟级） | **较高** | 50–200 张 | **YOLO26 首选** |
| `mmse` | 暴力搜索最小 MSE | 慢（小时级） | 最高 | 20–50 张 | 精度攻坚 |

> **建议路线**：`kl_divergence`（起点）→ 不达标 → `optimization_level=1` → 仍不达标 → `mmse` → 仍不达标 → 混合量化。

#### 2.5.3 `optimization_level` 是什么意思

RKNN 编译器在 `build()` 时对计算图做**算子融合**，把多个连续算子合并成一个"大算子"。融合的好处是中间结果留寄存器、省显存读写；代价是大算子只能共享一套量化参数 `(s, z)`。如果融合前的几个算子数值范围差异大，一个 scale 两头顾不好 → 精度掉。

```
原始图:     Conv → BN → ReLU → Conv
            ↑ 4个算子，各用各的 (s,z)，最精确

level=3:    FusedConvReLU → FusedConv
            ↑ 2个算子，共享 (s,z)，最快但可能顾此失彼
```

| 等级 | 融合程度 | YOLO26 风险 |
|------|---------|------------|
| 0 | 不融合，逐层独立量化 | 安全，最慢 |
| 1 | Conv+BN+ReLU 基础融合 | 较安全 |
| 2 (推荐起步) | 中级（算子重排、常量折叠） | 适中 |
| 3 (默认) | 最激进（跨层合并、布局转换） | O2M 头 Conv 可能被错误融合，输出值域偏移 → Sigmoid 全错 |

> **速度换精度的调参逻辑**：降低 level → 少融合 → 每个算子独立量化参数 → 更精确 → 但中间结果多落地几次 DDR → 更慢。YOLO26 建议从 **2** 起步。

### 2.6 混合量化（Hybrid Quantization）

```yaml
# YOLO26 O2M 模式的混合量化配置
customized_quantize_layers:
    # 检测头最后一层 Conv（输出 84 通道的那个），保持 FP16
    "/model.22/cv2.0/conv/Conv": float16
    "/model.22/cv3.0/conv/Conv": float16
    # 如有逐层分析标记的敏感层也加入

quantize_parameters:
    quantized_dtype: w8a8
```

```python
# 自动混合量化（实验性）
rknn.config(
    auto_hybrid_cos_thresh=0.98,
    optimization_level=2     # 配合降级
)
```

### 2.7 校准数据集构建

| 要素 | 建议 | 说明 |
|------|------|------|
| **数量** | **100–300 张** | INT8 量化 YOLO26 对校准集敏感 |
| **多样性** | 覆盖不同光照、角度、尺度、类别 | O2M 头每个尺度的激活值分布都要覆盖 |
| **来源** | 从 COCO val2017 均匀采样 | 与部署场景分布一致 |
| **预处理** | Resize 640×640, RGB, /255 | 与推理预处理严格一致 |

### 2.8 精度评估体系

| 指标 | 含义 | 本次目标 |
|------|------|----------|
| mAP@0.5 | IoU=0.5 时的 mAP | FP32 vs INT8 下降 < 1% |
| mAP@0.5:0.95 | 多 IoU 阈值平均 mAP | FP32 vs INT8 下降 < 1.5% |
| AP_small | 小目标 AP | YOLO26 的 STAL 对此有针对性优化，监控量化退化 |
| 有效检测数 | conf > 0.25 的检测框数量 | INT8 后不应骤降（骤降 = 输出头量化崩溃） |

### 2.9 RKNN C++ Runtime 推理原理

#### 2.9.1 标准 API 调用链

```
rknn_init()          → 加载模型，初始化 NPU context
rknn_query()         → 查询 I/O 属性和 SDK 版本
rknn_inputs_set()    → 设置输入图像数据
rknn_run()           → 执行 NPU 推理
rknn_outputs_get()   → 获取输出 (1, 84, 8400) float32
   ↓
CPU 后处理: sigmoid(conf) → NMS → 最终检测结果
   ↓
rknn_destroy()       → 释放所有资源
```

#### 2.9.2 性能测试指标

| 指标 | 获取方式 | 含义 |
|------|----------|------|
| **总推理时间** | `chrono` 测量全流程 | 端到端延迟 |
| **NPU 纯推理** | `rknn_query(PERF_DETAIL)` | 逐层 NPU 耗时 |
| **前处理** | resize + BGR2RGB | CPU 耗时 |
| **后处理** | sigmoid + NMS | CPU 耗时，YOLO26 传统模式的主要 CPU 开销 |
| **FPS** | 1000 / 总时延 | 吞吐量 |

---

## 三、代码示例与步骤

### 3.1 YOLO26 ONNX 导出

```python
# ============================================================
# 步骤1: YOLO26 → ONNX 导出 → 6 头图修改
# ============================================================
from ultralytics import YOLO

# 加载 YOLO26 预训练模型
model = YOLO("yolo26s.pt")

# 1.1 标准导出: end2end=False — 传统 O2M 输出
model.export(
    format="onnx",
    opset=12,
    imgsz=(640, 640),
    simplify=True,
    dynamic=False,
    batch=1,
    end2end=False                # ← 关键！
)
# 输出: yolo26s.onnx, shape (1, 84, 8400)

# 1.2 ONNX 图手术: 拆分为 6 头独立输出
# 在检测头 3 个 Conv 输出后，将 84 通道切为 reg(4ch) + cls(80ch)
# 生成 6 个命名输出节点：
#   output0_reg (1,4,6400), output0_cls (1,80,6400)
#   output1_reg (1,4,1600), output1_cls (1,80,1600)
#   output2_reg (1,4,400),  output2_cls (1,80,400)
# → 输出: yolov26s_6.onnx

# ---- 验证 ----
import onnx
m = onnx.load("yolov26s_6.onnx")
print("Input: ",  [(i.name, i.type.tensor_type.shape.dim) for i in m.graph.input])
print("Output:", [(o.name, o.type.tensor_type.shape.dim) for o in m.graph.output])
# 预期: Input  shape [1, 3, 640, 640]
#       Output 6 个节点，reg×3 + cls×3
```

**实现思路**：
- `end2end=False` 是关键——YOLO26 回退到传统 O2M 检测头，输出 `(1,84,8400)`。
- O2O 头（NMS-Free）在当前 RKNN Toolkit2 下 INT8 量化已知崩溃，本项目已验证。
- **6 头分离**是进一步优化：在 Concat 前将 84 通道拆为 reg(4ch) + cls(80ch) × 3 尺度，RKNN 为每个头独立分配 `(s,z)`。

**注意事项**：
- `end2end=True` 导出的 (1,300,6) 做 INT8 大概率输出全零。
- ONNX 图手术必须在 Concat 之前拆分，否则无法获得独立量化参数。
- 6 头 bbox 解码为 `(anchor - lt) * stride, (anchor + rb) * stride`，直接输出 xyxy，与标准 O2M 的 sigmoid decode 不同。

### 3.2 校准数据集准备

```python
# ============================================================
# 步骤2: 校准数据集准备
# ============================================================
import os, random

def prepare_calib(val_dir, output_txt, n=200):
    imgs = [os.path.join(val_dir, f) for f in os.listdir(val_dir)
            if f.lower().endswith(('.jpg','.png','.jpeg'))]
    sampled = random.sample(imgs, min(n, len(imgs)))
    with open(output_txt, 'w') as f:
        for p in sampled:
            f.write(p + '\n')
    print(f"[OK] {len(sampled)} images → {output_txt}")

prepare_calib("./coco_val2017/", "./calib_dataset.txt", n=200)

# 本项目实际: n=300 张（COCO val2017 采样），符合推荐范围。
# 见 calibration_data/data.py 的完整实现（重命名+生成路径清单）。
```

### 3.3 RKNN 量化转换

```python
# ============================================================
# 步骤3: ONNX → RKNN INT8 量化
# ============================================================
from rknn.api import RKNN

rknn = RKNN(verbose=True)

# 本项目实际配置（见 export_rknn_yolov26.py）
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    # 未显式设置: quantized_algorithm='normal' (默认)
    # 未显式设置: optimization_level=3 (默认)
    target_platform='rk3588'
)

ret = rknn.load_onnx(model='yolov26s_6.onnx')  # 6 头 ONNX
assert ret == 0, f"load_onnx fail: {ret}"

ret = rknn.build(
    do_quantization=True,           # 或 False → FP16
    dataset='./data.txt',           # 300 张校准图片
)
assert ret == 0, f"build fail: {ret}"

ret = rknn.export_rknn('yolo26s_6_rk3588_232_coco_int8.rknn')
assert ret == 0, f"export fail: {ret}"
print("[DONE]")
```

**注意事项**：
- 上述为项目实际配置（默认参数），实测 mAP@0.5 = 0.669。推荐按 5.4 节方案改进：`quantized_algorithm='kl_divergence'` + `optimization_level=2` + 校准集 200 张。
- 如果 build 阶段报 `[exDataConvert]` 相关错误，说明某层不支持该算子属性——尝试降低 `optimization_level=1`。

### 3.4 精度验证

```python
# ============================================================
# 步骤4: 精度分析 + mAP 验证
# ============================================================

# 4.1 逐层余弦相似度
perf = rknn.accuracy_analysis(
    inputs=['./calib_dataset.txt'],
    target='rk3588',
    device_id=None
)
for item in perf:
    c = item.get('cosine', 0)
    if c < 0.95:
        print(f"  ⚠️ LOW: {item['op_name']}  cosine={c:.4f}")
    else:
        print(f"  ✅ {item['op_name']}  cosine={c:.4f}")

# 4.2 板端推理 + COCO mAP
# 在板端跑 val2017 全部图片，保存 JSON 后用 pycocotools 计算
# （见 2.7 节 mAP 评估流程）
```

### 3.5 混合量化调优

```python
# ============================================================
# 步骤5: 混合量化（精度不达标时）
# ============================================================
rknn = RKNN(verbose=True)
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    quantized_dtype='w8a8',
    quantized_algorithm='mmse',           # 攻坚用 mmse
    optimization_level=2,
    target_platform='rk3588',
    auto_hybrid_cos_thresh=0.98
)
rknn.load_onnx(model='yolo26s.onnx')
rknn.build(do_quantization=True, dataset='./calib_dataset.txt')
rknn.export_rknn('yolo26s_hybrid_int8.rknn')
```

### 3.6 C++ RKNN 推理 + YOLO26 后处理 + 性能测试

```cpp
// ============================================================
// 步骤6: YOLO26 RKNN C++ 推理 (end2end=False 模式)
// (yolo26_rknn_infer.cpp)
// ============================================================
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "rknn_api.h"

// ---------- YOLO26 O2M 后处理 ----------
// 输入: output (1, 84, 8400) float32, 84=[x,y,w,h, cls0..cls79]
// 输出: 检测框列表 (NMS 后)
struct Box {
    float x1, y1, x2, y2, conf;
    int cls;
};

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

std::vector<Box> yolov8_postprocess(        // YOLO26 O2M 头格式与 YOLOv8 相同
    const float *output, int nc,            // nc = 80
    int img_w, int img_h,
    float conf_thr, float iou_thr)
{
    const int reg_max = 4, no = nc + reg_max;   // no = 84
    const int num_cells = 8400;
    const int strides[] = {8, 16, 32};
    const int grid_sizes[] = {80, 40, 20};

    std::vector<Box> dets;

    for (int ci = 0; ci < num_cells; ci++) {
        const float *p = output + ci * no;

        // 找 max class score
        float max_cls = 0;
        int   cls_id  = 0;
        for (int c = 0; c < nc; c++) {
            float s = sigmoid(p[reg_max + c]);
            if (s > max_cls) { max_cls = s; cls_id = c; }
        }
        if (max_cls < conf_thr) continue;

        // YOLO26 O2M 的 xywh 是相对于 cell 的偏移
        // 需要找到该 cell 属于哪个 stride/grid
        int cell_idx = ci;
        int stride = 0, grid = 0, offset = 0;
        for (int s = 0; s < 3; s++) {
            int gs = grid_sizes[s];
            int n   = gs * gs;
            if (cell_idx < offset + n) { stride = strides[s]; grid = gs; break; }
            offset += n;
        }

        int local = cell_idx - offset;
        int gy = local / grid, gx = local % grid;

        float x = (sigmoid(p[0]) * 2 - 0.5 + gx) * stride;
        float y = (sigmoid(p[1]) * 2 - 0.5 + gy) * stride;
        float w = powf(sigmoid(p[2]) * 2, 2) * stride;
        float h = powf(sigmoid(p[3]) * 2, 2) * stride;

        float x1 = x - w * 0.5f, y1 = y - h * 0.5f;
        float x2 = x + w * 0.5f, y2 = y + h * 0.5f;

        // 缩放回原图
        float sx = (float)img_w / 640.0f, sy = (float)img_h / 640.0f;
        x1 *= sx; y1 *= sy; x2 *= sx; y2 *= sy;
        x1 = std::max(0.0f, std::min(x1, (float)img_w));
        y1 = std::max(0.0f, std::min(y1, (float)img_h));
        x2 = std::max(0.0f, std::min(x2, (float)img_w));
        y2 = std::max(0.0f, std::min(y2, (float)img_h));

        dets.push_back({x1, y1, x2, y2, max_cls, cls_id});
    }

    // NMS
    std::sort(dets.begin(), dets.end(),
              [](const Box &a, const Box &b) { return a.conf > b.conf; });

    std::vector<Box> result;
    std::vector<bool> sup(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (sup[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (sup[j]) continue;
            float xx1 = std::max(dets[i].x1, dets[j].x1);
            float yy1 = std::max(dets[i].y1, dets[j].y1);
            float xx2 = std::min(dets[i].x2, dets[j].x2);
            float yy2 = std::min(dets[i].y2, dets[j].y2);
            float iw  = std::max(0.0f, xx2 - xx1);
            float ih  = std::max(0.0f, yy2 - yy1);
            float area_i = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1);
            float area_j = (dets[j].x2 - dets[j].x1) * (dets[j].y2 - dets[j].y1);
            float iou = (iw * ih) / (area_i + area_j - iw * ih + 1e-6f);
            if (iou > iou_thr) sup[j] = true;
        }
    }
    return result;
}

// ---------- 预处理 ----------
void preprocess(const cv::Mat &src, uint8_t *dst, int w, int h) {
    cv::Mat rgb, resized;
    cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(w, h));
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            cv::Vec3b &p = resized.at<cv::Vec3b>(y, x);
            dst[(y * w + x) * 3 + 0] = p[0];
            dst[(y * w + x) * 3 + 1] = p[1];
            dst[(y * w + x) * 3 + 2] = p[2];
        }
}

int main() {
    // ---- 1. 加载模型 ----
    const char *path = "yolo26s_int8.rknn";
    FILE *fp = fopen(path, "rb");
    fseek(fp, 0, SEEK_END); size_t sz = ftell(fp); fseek(fp, 0, SEEK_SET);
    unsigned char *md = (unsigned char *)malloc(sz);
    fread(md, 1, sz, fp); fclose(fp);

    // ---- 2. rknn_init ----
    rknn_context ctx;
    int ret = rknn_init(&ctx, md, sz, RKNN_FLAG_COLLECT_PERF_MASK, NULL);
    free(md);
    if (ret < 0) { printf("init fail %d\n", ret); return -1; }

    // ---- 3. 查询属性 ----
    rknn_input_output_num io;
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));

    rknn_tensor_attr ia = { .index = 0 }, oa = { .index = 0 };
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR,  &ia, sizeof(ia));
    rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &oa, sizeof(oa));
    printf("In: [%d,%d,%d,%d] Out: [%d,%d,%d]\n",
           ia.dims[0], ia.dims[1], ia.dims[2], ia.dims[3],
           oa.dims[0], oa.dims[1], oa.dims[2]);

    // ---- 4. 预热 ----
    uint8_t *ib = (uint8_t *)malloc(ia.size);
    memset(ib, 128, ia.size);
    rknn_input in = { 0, ib, (uint32_t)ia.size, RKNN_TENSOR_UINT8, RKNN_TENSOR_NHWC, 0 };
    rknn_inputs_set(ctx, 1, &in);
    rknn_run(ctx, NULL);
    rknn_output ow = { 0, 0, 0, NULL, 0 };
    rknn_outputs_get(ctx, 1, &ow, NULL);
    rknn_outputs_release(ctx, 1, &ow);

    // ---- 5. 正式测试 ----
    cv::Mat img = cv::imread("test.jpg");
    preprocess(img, ib, 640, 640);

    const int NW = 10, NT = 100;
    for (int i = 0; i < NW; i++) { rknn_inputs_set(ctx, 1, &in); rknn_run(ctx, NULL); }

    double total = 0;
    for (int i = 0; i < NT; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        rknn_inputs_set(ctx, 1, &in);
        rknn_run(ctx, NULL);
        rknn_output out = { 0, 1, 0, NULL, 0 };
        rknn_outputs_get(ctx, 1, &out, NULL);

        auto dets = yolov8_postprocess((float *)out.buf, 80,
                                        img.cols, img.rows, 0.25f, 0.45f);

        rknn_outputs_release(ctx, 1, &out);

        auto t1 = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    printf("\n========== YOLO26 INT8 性能 ==========\n");
    printf("  测试次数: %d\n", NT);
    printf("  平均延时: %.2f ms\n", total / NT);
    printf("  FPS:      %.2f\n", 1000.0 * NT / total);
    printf("========================================\n");

    // ---- 6. 逐层性能 ----
    rknn_perf_detail pd;
    rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, &pd, sizeof(pd));

    // ---- 7. 释放 ----
    free(ib);
    rknn_destroy(ctx);
    return 0;
}
```

**C++ 代码要点**：
1. **O2M 头后处理**：YOLO26 的 `end2end=False` 模式输出格式与 YOLOv8 一致——`(1, 84, 8400)`，后处理需要 sigmoid + cell 坐标映射 + NMS。
2. **xywh → xyxy**：`x = (sigmoid(px)*2 - 0.5 + grid_x) * stride`
3. **无 objectness**：YOLO26 的 class score 直接作为置信度，不需要乘 objectness。

### 3.7 编译与运行

```bash
# ============================================================
# 步骤7: 交叉编译 + 板端运行
# ============================================================
aarch64-linux-gnu-g++ -O3 -march=armv8.2-a \
    -I/path/to/rknn/runtime/include \
    -I/path/to/opencv-aarch64/include \
    yolo26_rknn_infer.cpp \
    -L/path/to/opencv-aarch64/lib \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
    -o yolo26_rknn_infer

scp yolo26_rknn_infer yolo26s_int8.rknn test.jpg user@rk3588:/tmp/
ssh user@rk3588 "cd /tmp && ./yolo26_rknn_infer"
```

### 3.8 量化优化报告

```python
# ============================================================
# 步骤8: 生成量化优化报告
# ============================================================
import json
from datetime import datetime

report = {
    "model": "YOLO26s (Ultralytics) + 6-Head Separation",
    "date": datetime.now().isoformat(),
    "platform": "RK3588",
    "strategy": "策略B+: 6头分离 + INT8 + CPU NMS",
    "onnx": "yolov26s_6.onnx (6 outputs: reg×3 + cls×3)",
    "config": {
        "quantized_dtype": "w8a8",
        "quantized_algorithm": "normal",      # 默认 MinMax（待优化为 kl_divergence）
        "optimization_level": 3,              # 默认最高融合（待降低为 2）
        "calib_images": 300
    },
    "accuracy": {
        "onnx_fp32_mAP50":    0.6848,
        "rknn_fp16_mAP50":    0.6836,
        "rknn_int8_mAP50":    0.6693,
        "int8_vs_fp32_drop": "-2.27%",
        "target_drop":        "< 1%",
        "pass":               False
    },
    "performance": {
        "int8_avg_ms":     "38.39ms（实测）",
        "fp16_avg_ms":     "76.97ms（实测）",
        "speedup_vs_fp16": "50.1%",
        "pass":            True
    },
    "notes": [
        "6头分离方案保证 INT8 不崩溃，但默认参数下精度降幅 2.27%，未达 1% 目标",
        "改进空间: kl_divergence + opt_level=2 + 校准集 200 张",
        "FP16 精度几乎无损（-0.18%），生产可直接使用"
    ],
    "conclusion": (
        "YOLO26s 6-Head INT8 (normal + opt_level=3 + 300 calib) "
        "在 RK3588 上 mAP@0.5 = 0.669（降幅 2.27%），推理 38.39ms（26 FPS），"
        "较 FP16 提速 50.1%。精度未达标，后续可尝试 kl_divergence + optimization_level=2 + 混合量化等方案。"
    )
}

with open("quantization_report.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(report["conclusion"])
```

---

## 四、学习资料整理

### 4.1 核心速查表

| 类别 | 项目 | 内容 |
|------|------|------|
| **模型** | 来源 | Ultralytics `yolo26s.pt`, `ultralytics≥8.4.0` |
| **模型** | 端到端输出 | `(1, 300, 6)` xyxy+conf+cls, NMS-Free |
| **模型** | 传统输出 | `(1, 84, 8400)` xywh+80cls, 需要 NMS |
| **模型** | 6头分离输出 | 6 节点: `output{0,1,2}_{reg,cls}`, reg=4ch lt/rb 偏移 |
| **导出** | 稳定模式 | `end2end=False, opset=12, simplify=True` → ONNX 图手术拆分 6 头 |
| **导出** | 风险模式 | `end2end=True` → INT8 已知崩溃 |
| **量化** | 优化等级 | **从 `optimization_level=2` 起步**（默认 3 对 YOLO26 过激进） |
| **量化** | 推荐算法 | `kl_divergence`（当前用 normal，待优化） |
| **量化** | 校准集 | **100–300 张**（当前 300 张，精度主要瓶颈） |
| **后处理** | 标准O2M | sigmoid + 三尺度 grid decode + NMS |
| **后处理** | 6头方案 | `(anchor-lt)*stride, (anchor+rb)*stride` → xyxy + NMS |
| **性能** | 性能采集 | `RKNN_FLAG_COLLECT_PERF_MASK` |

### 4.2 必背核心结论

1. **YOLO26 的 O2O 端到端头目前不能做 INT8 量化** — 输出 (1,300,6) 在 RKNN Toolkit2 当前版本下 INT8 量化后检测崩溃，这是已知社区问题（GitHub Issue #23753）。
2. **6头分离 + INT8 是本项目实际方案** — ONNX 图手术拆分为 reg/cls 独立输出，RKNN 为每个头分配独立 `(s,z)`，INT8 量化稳定不崩溃。实测 mAP@0.5 = 0.669（降幅 2.27%），精度未达 1% 目标但有明确优化路径。
3. **`optimization_level` 必须从 2 起步** — YOLO26 的新架构对图优化更敏感，默认的 level=3 激进融合可能破坏输出层。本项目用 level=3 未崩溃（得益于 6 头分离），但精度或可再提升。
4. **YOLO26 后处理无 DFL、无 objectness** — 相比 YOLOv8 少了 DFL、相比 YOLOv6 少了 objectness。6 头方案的后处理为 `(anchor-lt)*stride` 直接得 xyxy，比标准 O2M 的 sigmoid decode 更简洁。
5. **FP16 是生产安全选项** — 实测 FP16 精度几乎无损（mAP 下降 < 0.2%），可直接上线。INT8（38.39ms）较 FP16（76.97ms）提速 50.1%。
6. **校准集质量与算法选择是精度关键** — 当前 300 张 COCO val2017 采样（数量符合推荐范围），但使用默认 `normal` (MinMax) 算法。切换 `kl_divergence` 并降低 `optimization_level` 是后续精度优化的主要方向。

### 4.3 YOLO26 与同类模型部署对比

| 对比维度 | YOLO26 (end2end=True) | YOLO26 (end2end=False) | **YOLO26 (6头分离)** | YOLOv8 | YOLOv6 |
|----------|----------------------|------------------------|---------------------|--------|--------|
| **输出格式** | (1,300,6) xyxy | (1,84,8400) xywh | 6 节点: reg×3+cls×3 | 3 个 head | (1,8400,85) |
| **DFL** | 无 | 无 | 无 | 有 | 无 |
| **Objectness** | 无 | 无 | 无 | 无 | 有 |
| **NMS** | **不需要** | 需要 | 需要 | 需要 | 需要 |
| **bbox 解码** | 直接 xyxy | sigmoid(xywh) | anchor±lt/rb | DFL+sigmoid | sigmoid(xywh) |
| **RKNN INT8** | ❌ 崩溃 | ✅ 可用 | ✅ **稳定** (实测 0.669) | ✅ 成熟 | ✅ 可用 |
| **RKNN FP16** | ✅ 推荐 | ✅ 可用 | ✅ 推荐 (实测 0.684) | ✅ 可用 | ✅ 可用 |

### 4.4 完整工作流总览

```
┌─────────────────────────────────────────────────────────────┐
│  ① YOLO26.pt → ONNX (end2end=False, opset=12)               │
│      ↓                                                       │
│  ② ONNX 图手术 → 拆分为 6 头独立输出 (reg×3 + cls×3)         │
│      ↓                                                       │
│  ③ Netron 验证: 6 个 output 节点维度正确                      │
│      ↓                                                       │
│  ④ 校准数据集 (推荐 100-300 张 COCO val，当前项目 300 张)      │
│      ↓                                                       │
│  ⑤ rknn.config(optimization_level=2, algorithm='kl_divergence') │
│     → load_onnx('yolov26s_6.onnx') → build(do_quantization=True) │
│      ↓                                                       │
│  ⑥ 逐层 cosine 分析 → 定位敏感层                              │
│      ↓ (cosine<0.95 的层 < 5%)   ↓ (cosine<0.95 的层 很多)   │
│  ⑦ 导出 .rknn                   ⑦ 混合量化 / 降 opt_level    │
│      ↓                                                       │
│  ⑧ C++/Python 推理: 6 头解码 + NMS + 性能测试                 │
│      ↓                                                       │
│  ⑨ 生成量化报告 → 验证 mAP<1% & speedup>40%                  │
└─────────────────────────────────────────────────────────────┘
```

### 4.5 常见问题解答 (FAQ)

**Q1: 为什么 YOLO26 的端到端头 (end2end=True) 不能做 INT8？**

O2O 头权重稀疏且输出通道极少（6 通道），INT8 量化后信息丢失 → 300 个输出全部置信度 < 0.01 → 检测崩溃。社区 Issue #23753 跟踪此问题，等待 RKNN Toolkit2 更新。

**Q2: `end2end=False` 就一定要在 CPU 做 NMS，会不会太慢？**

8400 个 cell 的 NMS 在 ARM Cortex-A76 上通常 < 2ms。加上 sigmoid 计算，总后处理 < 3ms。相比 NPU 推理时间（实测 38.39ms），占比 < 8%，可接受。

**Q3: `optimization_level=2` 和 `3` 的精度差多少？**

实测差异因模型而异。对 YOLO26，level=3 的激进融合可能在特定场景下破坏检测头 Sigmoid。level=2 更安全。（未在本项目中做 level=2 vs level=3 对照实验，结论来自社区经验）

**Q4: YOLO26 的 FP16 RKNN 性能如何？能不能不折腾 INT8？**

YOLO26n FP16 在 RK3588 上的官方基准为 65.7ms（15 FPS），YOLO26s FP16 官方基准 99.2ms（10 FPS）。本项目实测 YOLO26s 6 头 FP16 为 76.97ms（13 FPS），6 头 INT8 为 38.39ms（26 FPS），提速 50.1%，超过 40% 目标。

**Q5: 怎么快速判断 INT8 量化是否崩溃？**

跑一张测试图片，检查：模型输出的 (1, 84, 8400) 中 sigmoid 后的 max confidence 是否 > 0.2。如果全部 < 0.01，量化崩溃。

**Q6: 校准集数量对 YOLO26 有多大影响？**

YOLO26 的 O2M 头三个尺度激活值分布不同，校准集过少会导致某尺度量化不准，增加校准集数量是提升量化精度最直接的手段。具体数值因模型和数据集而异，本项目中未做校准集数量的消融实验。

**Q7: 如果一切调优都失败了怎么办？**

(1) 回退 FP16（精度无损，实测 76.97ms / 13 FPS）；(2) 换用 YOLOv8（RKNN INT8 生态最成熟）；(3) 等待 RKNN Toolkit2 对 YOLO26 O2O 头的 INT8 修复。

### 4.6 推荐学习路径

| 阶段 | 内容 | 时间 |
|------|------|------|
| **理论基础** | 2.1–2.3 节：PTQ + YOLO26 架构 + INT8 已知问题 | 1.5h |
| **动手(上)** | 3.1–3.3 节：ONNX 导出 → RKNN 量化 | 1h |
| **动手(下)** | 3.4–3.5 节：精度分析 + 混合量化调优 | 1h |
| **C++ 部署** | 3.6–3.7 节：后处理 + 性能测试 | 1.5h |
| **报告输出** | 3.8 节 | 0.5h |
| **延伸** | Ultralytics RKNN 集成文档 + rknn_model_zoo YOLO 示例 | 1h |

---

## 五、实践案例：YOLO_Tooklits 项目的 6 头分离量化方案

> 来源：`E:\A_HHD\YOLO_Tooklits`，基于 YOLO26s + RK3588 的实际工程项目。

### 5.1 核心思路：回归/分类头分离

标准 YOLO26 的 `end2end=False` 输出是一个 `(1, 84, 8400)` 的张量——回归 (4ch) 和分类 (80ch) 共用一套量化参数。本项目在 ONNX 导出后对计算图做修改，将 84 通道按 `[0:4]` 和 `[4:84]` 切分，拆为 **6 个独立输出节点**：

```
输出节点                 尺度       通道    内容
output0_reg             80×80      4       lt_x, lt_y, rb_x, rb_y（左上/右下偏移）
output0_cls             80×80      80      类别 logits
output1_reg             40×40      4       lt_x, lt_y, rb_x, rb_y
output1_cls             40×40      80      类别 logits
output2_reg             20×20      4       lt_x, lt_y, rb_x, rb_y
output2_cls             20×20      80      类别 logits
```

**为什么这样做？** 回归和分类的数值分布差异巨大——坐标偏移量范围小、精度敏感；分类 logits 范围大、激活后压缩。共用一个 scale 时量化参数顾此失彼。分离后 RKNN 为 6 个头各分配独立的 `(s, z)`，回归不拖累分类，分类不污染回归。

### 5.2 实现方法

#### 5.2.1 ONNX 导出与图修改

```python
# 步骤1: Ultralytics 标准导出（end2end=False, opset=12）
model = YOLO("yolo26s.pt")
model.export(format="onnx", opset=12, imgsz=(640,640),
             simplify=True, dynamic=False, batch=1, end2end=False)
# → 输出 yolo26s.onnx, shape (1, 84, 8400)

# 步骤2: ONNX 图手术 — 拆分 84 通道为 reg(4ch) + cls(80ch) × 3 尺度
# 在检测头的 3 个 Conv 输出后插入 Split 节点, 生成 6 个命名输出
# → 输出 yolov26s_6.onnx
```

#### 5.2.2 RKNN 量化转换

```python
# 出自 export_rknn_yolov26.py（完整脚本见项目根目录）
from rknn.api import RKNN

rknn = RKNN(verbose=True)
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rk3588',
)
rknn.load_onnx(model='yolov26s_6.onnx')
rknn.build(do_quantization=True, dataset='./data.txt')   # INT8
# rknn.build(do_quantization=False)                      # FP16
rknn.export_rknn('yolo26s_6_rk3588_232_coco_int8.rknn')
```

#### 5.2.3 后处理解码

6 头方案的 bbox 输出是 **lt/rb 角点偏移**，非传统 xywh。解码在 `PostProcessor_for_YOLOv26._process_rknn_6_heads()` 中实现：

```python
# 出自 YOLO/blocks/postprocessor.py:1141-1196
# 1. 分离 6 个输出 → reg(4ch) 和 cls(80ch)
# 2. 拼接 3 个尺度: final_regs (1,4,8400), final_clss (1,80,8400)
# 3. 解码 bbox（关键差异：lt/rb 偏移而非 xywh 中心偏移）
lt  = final_regs[:, :2, :]          # 预测点到左上角偏移
rb  = final_regs[:, 2:, :]          # 预测点到右下角偏移
x1y1 = (anchor_points - lt) * stride_tensor
x2y2 = (anchor_points + rb) * stride_tensor
boxes = np.concatenate([x1y1, x2y2], axis=1)  # 直接 xyxy，无需转换
# 4. sigmoid(cls) → conf 过滤 → 类别独立 NMS → reverse_letterbox
```

> **与教案 C++ 后处理的差异**：教案用 xywh + sigmoid decode（`sigmoid(px)*2-0.5`），本项目用 lt/rb 偏移 + anchor 直接计算 xyxy，无需求 sigmoid 于 bbox 部分。

### 5.3 精度验证结果

**测试条件**：COCO val2017 全部 5000 张，pycocotools 评估，conf=0.001, iou=0.6。

| 模型 | mAP@0.5 | mAP@0.5:0.95 | ΔmAP@0.5 | ΔmAP@0.5:0.95 |
|------|---------|:---:|:---:|:---:|
| ONNX FP32 (基线) | 0.6848 | 0.5261 | — | — |
| RKNN FP16 | 0.6836 | 0.5252 | -0.18% | -0.17% |
| RKNN INT8 | 0.6693 | 0.5003 | **-2.27%** | **-4.91%** |

**结论**：
- FP16 几乎无损（下降 < 0.2%），可直接用于生产
- INT8 mAP@0.5 下降 2.27%，**未达到教案 1% 目标**（差距约 1.3 个百分点）
- INT8 mAP@0.5:0.95 下降 4.91%，说明高 IoU 定位精度受 INT8 影响更大
- 6 头分离方案保证了 INT8 量化**不崩溃**（无全零输出），但精度仍有提升空间

#### 5.3.1 推理性能（板端实测）

数据来源：`E:\A_HHD\yolov26_deploy\README.md`，RK3588 C++ 推理，100 次平均，不含前后处理。

| 模型 | 量化 | 推理耗时 | FPS |
|------|:---:|:---:|:---:|
| 6头分离 | INT8 | **38.39 ms** | 26.0 |
| 6头分离 | FP16 | **76.97 ms** | 13.0 |
| 官方单输出 | FP16 | **80.07 ms** | 12.5 |
| 官方单输出 | INT8 | **42.56 ms** | 23.5 |

**结论**：
- 6 头 INT8 较 6 头 FP16 提速 **50.1%**，较官方单输出 FP16 提速 **52.1%**，超过 40% 目标
- 6 头分离比官方单输出 INT8 快 4.2ms（38.39 vs 42.56），分离量化对 NPU 执行效率有正向收益
- 1.6.0 版本 toolkit 转换的模型速度慢（129.5ms），**必须用 2.3.2 版本转换**才能获得正常性能

---

*实践案例基于https://github.com/kogap1/YOLO_Tookit项目真实代码与数据（2026-06-16）。*
