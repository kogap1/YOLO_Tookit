# -*- coding: utf-8 -*-
"""
@File    : main_for_yolov26.py
@Time    : 2026/02/27
@Version : 实验版本
@Description: YOLOv26 推理入口脚本
"""

import argparse
from YOLO.blocks.pipeline import YOLOv26

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="onnx or rknn model", default="YOLO/models/yolov26/yolov26s.onnx")
    parser.add_argument("--source", type=str, help="source file or dir", default="test_img")
    parser.add_argument("--show", action="store_true", help="show video when GUI")
    parser.add_argument("--track", action="store_true", help="use multi-object tracking")
    parser.add_argument("--output", type=str, help="output file or dir", default="results")

    args = parser.parse_args()

    # 实例化 pipeline
    pipeline = YOLOv26(
        model_path=args.model,
        max_queue_size=50,
        model_parallel=12,
        nms_thres=0.5,
        conf_thres=0.4,
        show=args.show,
        use_track=args.track
    )

    # 启动推理
    pipeline.run(args.source, args.output)