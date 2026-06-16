# YOLOv26 推理部署

基于 YOLO 通用推理框架的 YOLOv26 模型部署方案，支持 ONNX / RKNN 多后端推理。

## 快速开始

### 环境配置

```bash
cd YOLO_Toolkits
conda create -n yolotk python=3.12 -y
conda activate yolotk
pip install .
```

- Mac 平台自动安装 `coremltools==9.0`
- Rockchip 平台自动安装 `rknn-toolkit-lite2==2.3.2`

### ONNX 推理

```bash
# 单输出头 (output=1)
python main_for_yolov26.py --model models/yolov26/yolo26s_1.onnx --source test_img

# 6 输出头 (output=6，适配 RKNN 导出)
python main_for_yolov26.py --model models/yolov26/yolo26s_6.onnx --source test_img

# 端到端 NMS-Free 输出
python main_for_yolov26.py --model models/yolov26/yolo26s_end2end.onnx --source test_img
```

### RKNN 推理 (RK3588)

```bash
# INT8 量化模型
python main_for_yolov26.py --model models/yolov26/yolo26s_6_rk3588_int8.rknn --source test_img

# FP16 模型
python main_for_yolov26.py --model models/yolov26/yolo26s_6_rk3588_fp16.rknn --source test_img
```

## 模型转换

### .pt → .onnx

```bash
yolo export model=yolo26s.pt format=rknn
```

导出后使用 Netron 查看，确认 6 个输出头即为正常。

### .onnx → .rknn

```bash
# INT8 量化
python export_rknn_yolov26.py --model yolo26s_6.onnx --data data.txt --int8 --output yolo26s_6_rk3588

# FP16
python export_rknn_yolov26.py --model yolo26s_6.onnx --data data.txt --output yolo26s_6_rk3588
```

量化数据集 `data.txt` 格式为每行一张图片路径，建议 200 张以上。

## RK3588 推理性能

| 模型 | 量化 | 耗时 (100次平均) | FPS | mAP@50 | mAP@50:95 |
|------|------|-----------------|-----|--------|-----------|
| yolo26s_6 (自定义6头) | INT8 | 38.4 ms | 26.0 | 0.669 | 0.500 |
| yolo26s_6 (自定义6头) | FP16 | 77.0 ms | 13.0 | 0.684 | 0.525 |
| ONNX (FP32 基准) | — | — | — | 0.685 | 0.526 |
| yolo26s (官方导出) | FP16 | 80.1 ms | 12.5 | — | — |
| yolo26s (官方导出 + 自转) | INT8 | 42.6 ms | 23.5 | — | — |

> 测试平台：RK3588，NPU 自动分配核心。mAP 在 COCO val2017 上测试。

## RK3588 板端环境

```bash
conda create -n rknn_py310 python=3.10 -y
conda activate rknn_py310

# 基础依赖
pip install numpy opencv-python onnx==1.14.0 onnxruntime==1.14.0 pyyaml pillow

# RKNN Toolkit Lite2
# 下载对应版本：https://github.com/airockchip/rknn-toolkit2
pip install rknn_toolkit_lite2-*-cp310-cp310-linux_aarch64.whl
```

## 项目结构

```
├── main_for_yolov26.py        # YOLOv26 推理入口
├── export_rknn_yolov26.py     # ONNX → RKNN 导出工具
├── YOLO/
│   └── blocks/
│       ├── pipeline.py         # 推理流水线 (BasePipeline, YOLOv26)
│       ├── inferencer.py       # 多后端推理 (ONNX/RKNN/CoreML)
│       ├── preprocessor.py     # 图像预处理
│       ├── postprocessor.py    # 后处理 (NMS/DFL/坐标还原)
│       ├── visualizer.py       # 可视化
│       └── tracker/            # ByteTrack 多目标跟踪
└── setup.py
```

## 参考

- [Ultralytics YOLOv26 Rockchip RKNN 导出](https://docs.ultralytics.com/zh/integrations/rockchip-rknn/)
- [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
