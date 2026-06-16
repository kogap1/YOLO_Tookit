import os
import json
import argparse
import numpy as np
from rknn.api import RKNN


def opt():
    parser = argparse.ArgumentParser(description="YOLOv26s 6-Head RKNN Converter")
    parser.add_argument("--model", type=str, required=True, help='input yolo26s_6.onnx path')
    parser.add_argument("--batch_size", type=int, default=1, help='batch size')
    parser.add_argument("--data_path", type=str, default='./dataset.txt', help='quantize dataset')
    parser.add_argument("--output", type=str, default='yolo26s_6_rk3588.rknn', help='output path')
    parser.add_argument("--int8", action="store_true", help='True: int8, False: fp16')
    return parser.parse_args()


def build_rknn(opt):
    rknn = RKNN(verbose=True)

    # 1. 精确定义 6 头输出节点名称 (严格对应 Netron 截图)
    # 顺序：80x80 (reg, cls) -> 40x40 (reg, cls) -> 20x20 (reg, cls)
    output_nodes = [
        'output0_reg', 'output0_cls',
        'output1_reg', 'output1_cls',
        'output2_reg', 'output2_cls'
    ]

    # 2. 配置 RKNN 环境
    print("--> 配置 RKNN 环境...")
    # 修正：output_optimize 必须为 True，不能是 1
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform='rk3588',
    )

    # 3. 加载模型
    print(f"--> 加载模型: {opt.model}")
    ret = rknn.load_onnx(model=opt.model)
    if ret != 0:
        print('Load ONNX failed!');
        exit(ret)

    # 4. 构建与量化
    print(f"--> 开始构建 (INT8={opt.int8})...")
    ret = rknn.build(
        do_quantization=opt.int8,
        dataset=opt.data_path,
        rknn_batch_size=opt.batch_size
    )
    if ret != 0:
        print('Build RKNN failed!');
        exit(ret)

    # 5. 导出
    rknn_name = opt.output.replace(".rknn", "_int8.rknn") if opt.int8 else opt.output.replace(".rknn", "_fp16.rknn")
    ret = rknn.export_rknn(rknn_name)
    if ret != 0:
        print('Export RKNN failed!');
        exit(ret)
    print(f"--> 成功导出: {rknn_name}")

    # 6. 仿真推理 (用于确认输出维度)
    rknn.init_runtime()
    # 模拟 C++ 端的输入数据排列
    dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    outputs = rknn.inference(inputs=[dummy_input], data_format='nhwc')

    print("\n" + "=" * 20 + " 验证输出维度 " + "=" * 20)
    for i, out in enumerate(outputs):
        print(f"节点 [{output_nodes[i]}] 形状: {out.shape}")

    rknn.release()


if __name__ == '__main__':
    args = opt()
    build_rknn(args)
