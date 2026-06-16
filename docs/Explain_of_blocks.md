# 模块说明

## 前处理

* preprocessor.py

  * BasePreProcessor

    * `__init__(input_size=(640, 640))`
      初始化前处理模块，设置模型输入尺寸

    * `letterbox(im, new_shape, color)`
      等比例缩放并进行 padding
      返回 `(image, scale_rate, (dw, dh))`

    * `__call__(img)`
      前处理抽象接口（提供默认实现）

  * PreProcessor_for_YOLO_general(BasePreProcessor)

    * `__init__(input_size)`
      初始化通用 YOLO 前处理模块

    * `__call__(image_path)`
      执行标准 YOLO 前处理流程（完全继承 BasePreProcessor）

## 推理

* inferencer.py

  * Inferencer

    * `__init__(model_path, rknn_core="auto")`
      初始化推理器，支持 ONNX / RKNN / CoreML 模型
    * `load_model()`
      加载模型文件并初始化对应推理后端
    * `get_info()`
      获取并打印模型输入输出信息
    * `init_runtime()`
      初始化 RKNN 运行时（仅 `.rknn` 模型需要）
    * `norm(input_data)`
      根据模型类型对输入数据进行归一化处理
    * `infer(input_data)`
      执行模型推理，返回模型输出结果
    * `release()`
      释放推理资源（RKNN 专用）

## 后处理

* postprocessor.py

  * BasePostProcessor

    * `__init__(input_size, conf_thres, nms_thres, strides, feats_hw)`
      初始化后处理基础参数（输入尺寸、阈值、特征层配置）
    * `init_anchors(feats_shapes, strides)`
      生成 YOLO anchor points 与 stride 张量
    * `nms(boxes, scores, class_ids, iou_threshold, xywh)`
      多类别、m:n IoU 向量化 NMS
    * `sigmoid(x)`
      数值稳定的 Sigmoid 函数
    * `xywh_to_xyxy(boxes)`
      边框格式转换：xywh → xyxy
    * `bbox_iou(box1, box2, xywh, GIoU, DIoU, CIoU)`
      IoU / GIoU / DIoU / CIoU 计算（支持批量）
    * `reverse_letterbox(boxes, r, dwdh)`
      将检测框从 letterbox 坐标系还原到原图
    * `reverse_scale_pose(pose, r, dwdh)`
      将姿态关键点从 letterbox 坐标系还原到原图
    * `dfl(position)`
      DFL（Distribution Focal Loss）解码边界框偏移
    * `standard_outputs_postprocess(outputs, scale_rate, dwdh, pose_model)`
      标准模型输出后处理（检测 / 姿态通用）
    * `__call__(outputs, scale_rate, dwdh)`
      抽象接口，子类实现具体后处理逻辑

  * PostProcessor_for_YOLOv5(BasePostProcessor)
  
  * PostProcessor_for_YOLOv8(BasePostProcessor)
  
  * PostProcessor_for_RopeSkipping(BasePostProcessor)
  
  * PostProcessor_for_YOLOv8_Face(BasePostProcessor)

## 可视化

* visualizer.py

  * Visualizer

    * `__init__(class_names=None, show_boxes=True, show_points=True)`
      初始化可视化模块
    * `_generate_colors(num_classes)`
      为每个类别生成 RGB 颜色
    * `visualize(img, boxes, scores, cls, pose, save_path=None)`
      可视化检测框和姿态关键点


## pipeline

* pipeline.py

  * BasePipeline

    * `__init__(...)` 初始化推理流水线基础配置
    * `__enter__()` 加载模型并初始化运行时
    * `__exit__(...)` 释放模型资源
    * `_run_image(image_path, save_path=None)` 单张图片推理（不释放模型）
    * `_run_video(video_path, save_path=None)` 同步视频推理
    * `_run_video_async(video_path, save_path=None, ...)` 异步多模型并行视频推理
    * `plugin(boxes, scores, cls, pose, src_img, image_path)` 子类可重写的业务钩子
    * `run_image(image_path, save_path=None)` 图片推理（自动管理模型生命周期）
    * `run_video(video_path, save_path=None)` 视频推理（异步）
    * `run_file(file_path, save_dir)` 自动识别文件类型并推理
    * `run(file_path, save_dir=None)` 最高级接口，支持文件 / 文件夹 / 摄像头 / RTSP

  * YOLOv5(BasePipeline)

  * YOLOv8(BasePipeline)

  * YOLOv8_Face(BasePipeline)

  * RopeSkippingPose(BasePipeline)







