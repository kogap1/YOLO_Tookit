import numpy as np
from typing import Literal, Tuple, Optional
from abc import ABC, abstractmethod
from typing import Literal, Tuple, Optional, List, Dict, Any, Union

class BasePostProcessor(ABC):
    
    def __init__(self,
                 input_size:tuple=(640, 640),
                 conf_thres:float=0.2,
                 nms_thres:float=0.5,
                 strides:list=[8, 16, 32],
                 feats_hw:list=[(80, 80), (40, 40), (20, 20)],
                 ):
        """
        BasePostProcessor 初始化方法，用于设置模型输入和后处理参数。

        Args:
            input_size (tuple, optional): 模型输入图像的尺寸 (height, width)。默认值为 (640, 640)。
            conf_thres (float, optional): 置信度阈值，低于该阈值的检测框将被过滤。默认值为 0.2。
            nms_thres (float, optional): 非极大值抑制 (NMS) 的 IoU 阈值。默认值为 0.5。
            strides (list, optional): YOLO 输出特征图对应的下采样步幅 (stride)，
                                    与 feats_hw 一一对应。默认值为 [8, 16, 32]。
            feats_hw (list, optional): YOLO 输出特征图的高宽 (height, width)，与 strides 对应。
                                    默认值为 [(80, 80), (40, 40), (20, 20)]。
        """

        self.input_size = input_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.strides = strides
        self.feats_hw = feats_hw


    def init_anchors(self,
                     feats_shapes:list=[(80, 80), (40, 40), (20, 20)],
                     strides:list=[8, 16, 32],
                     ):
        """
        生成anchor和stride

        Args:
            feats_shapes (list, optional): yolo三个输出的特征尺寸. Defaults to [(80, 80), (40, 40), (20, 20)].
            strides (list, optional): yolo三个输出的步长. Defaults to [8, 16, 32].

        Returns:
            tuple: (anchor, stride)
        """
        
        anchor_points, stride_tensor = [], []
        for (h, w), stride in zip(feats_shapes, strides):
            sx = np.arange(w) + 0.5
            sy = np.arange(h) + 0.5
            sy, sx = np.meshgrid(sy, sx, indexing='ij')
            anchors = np.stack((sx, sy), axis=-1).reshape(-1, 2)
            anchor_points.append(anchors)
            stride_tensor.append(np.full((h * w, 1), stride, dtype=np.float32))
        return np.expand_dims(np.vstack(anchor_points).T, axis=0), np.vstack(stride_tensor).T


    def nms(self,
            boxes: np.ndarray,
            scores: np.ndarray,
            class_ids: np.ndarray,
            iou_threshold=0.9,
            xywh=True,
            ):
        """
        NMS过滤 - 支持m:n计算iou

        Args:
            boxes (np.ndarray): 检测框，形状为[num_of_boxes, 4]。
            scores (np.ndarray): 每个检测框的得分，形状为[num_of_boxes,]
            class_ids (np.ndarray): 每个检测框的类别，形状为[num_of_boxes,]
            iou_threshold (float): NMS的iou阈值. Defaults to 0.5.
            xywh (bool): 输入boxes的格式是否为xywh，True为xywh，False为xyxy。 Defaults to True.

        Returns:
            list: 最终过滤后保留下来的检测框的索引，0 <= len(list) <= num_of_boxes
        """
        
        keep = []
        unique_classes = np.unique(class_ids)

        # 类别独立处理
        for cls in unique_classes:
            mask = class_ids == cls
            cls_boxes = boxes[mask]
            cls_scores = scores[mask]
            cls_indices = np.where(mask)[0]

            # 排序
            order = cls_scores.argsort()[::-1]

            while order.size > 0:
                # 3. 选择当前分数最高的框
                i = order[0]
                keep.append(cls_indices[i])

                if order.size == 1:
                    break

                # ----------- m:n 的关键 --------------
                # 当前最高分框与剩余所有框一起做 IoU 向量化计算
                current = cls_boxes[i:i+1]      # shape (1,4)
                others = cls_boxes[order[1:]]   # shape (n,4)

                # IoU shape (1,n) → squeeze 成 (n,)
                ious = self.bbox_iou(current, others, xywh=xywh).reshape(-1)

                # 过滤掉 IoU 过大的框
                remain_mask = ious < iou_threshold

                # 更新剩余框索引
                order = order[1:][remain_mask]

        return keep


    def sigmoid(self,
                x:np.ndarray):
        """
        计算数值稳定的 Sigmoid 函数，将输入映射到 [0, 1] 区间。

        Args:
            x (np.ndarray): 输入的数值或数组。

        Returns:
            np.ndarray: 对应输入的 Sigmoid 值，范围在 (0, 1) 之间。

        Notes:
            - 为防止指数溢出或下溢，输入 x 会被限制在 [-15, 15]。
            - Sigmoid 公式：
                σ(x) = 1 / (1 + exp(-x))
            - 当 x > 15 时，σ(x) ≈ 0.9999997；当 x < -15 时，σ(x) ≈ 3.059e-7。
        """
        x = np.clip(x, -15, 15)
        return 1 / (1 + np.exp(-x))


    def xywh_to_xyxy(self,
                     boxes:np.ndarray):
        """
        如下转化: [x_center, y_center, w, h] -> [x1, y1, x2, y2]

        Args:
            boxes (np.ndarray): xywh格式的检测框（x、y为检测框中心坐标，w、h为检测框宽高）

        Returns:
            np.ndarray: xyxy格式的检测框（x1y1x2y2分别代表左上角坐标和右下角坐标）
        """
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)


    def bbox_iou(self,
                 box1:np.ndarray,
                 box2:np.ndarray,
                 xywh:bool=True,
                 GIoU:bool=False,
                 DIoU:bool=False,
                 CIoU:bool=False,
                 eps:float=1e-7,
                 ):
        """
        计算 IoU / GIoU / DIoU / CIoU，支持 NumPy 数据。

        Args:
            box1 (np.ndarray): shape (m,4) 或 (4,) 单个或多个框
            box2 (np.ndarray): shape (n,4) 或 (4,) 单个或多个框
            xywh (bool): True 表示输入格式 (x, y, w, h)，False 表示 (x1, y1, x2, y2)
            GIoU (bool): 是否计算 GIoU
            DIoU (bool): 是否计算 DIoU
            CIoU (bool): 是否计算 CIoU
            eps (float): 防止除零

        Returns:
            np.ndarray: shape (m, n)，每个 box1 对 box2 的 IoU 或其他指标
        """
        # 确保 box1, box2 维度为 (m,4), (n,4)
        box1 = np.atleast_2d(box1)
        box2 = np.atleast_2d(box2)

        if xywh:
            # xywh -> xyxy
            b1_x1 = box1[:, 0:1] - box1[:, 2:3]/2
            b1_y1 = box1[:, 1:2] - box1[:, 3:4]/2
            b1_x2 = box1[:, 0:1] + box1[:, 2:3]/2
            b1_y2 = box1[:, 1:2] + box1[:, 3:4]/2

            b2_x1 = box2[:, 0:1].T - box2[:, 2:3].T/2
            b2_y1 = box2[:, 1:2].T - box2[:, 3:4].T/2
            b2_x2 = box2[:, 0:1].T + box2[:, 2:3].T/2
            b2_y2 = box2[:, 1:2].T + box2[:, 3:4].T/2
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0:1], box1[:, 1:2], box1[:, 2:3], box1[:, 3:4]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0:1].T, box2[:, 1:2].T, box2[:, 2:3].T, box2[:, 3:4].T

        # intersection
        inter_w = np.clip(np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1), 0, None)
        inter_h = np.clip(np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1), 0, None)
        inter_area = inter_w * inter_h

        # union
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union = area1 + area2 - inter_area + eps

        iou = inter_area / union

        if GIoU or DIoU or CIoU:
            # convex box
            c_x1 = np.minimum(b1_x1, b2_x1)
            c_y1 = np.minimum(b1_y1, b2_y1)
            c_x2 = np.maximum(b1_x2, b2_x2)
            c_y2 = np.maximum(b1_y2, b2_y2)
            c_w = c_x2 - c_x1
            c_h = c_y2 - c_y1

            if DIoU or CIoU:
                # center distance
                b1_cx = (b1_x1 + b1_x2) / 2
                b1_cy = (b1_y1 + b1_y2) / 2
                b2_cx = (b2_x1 + b2_x2) / 2
                b2_cy = (b2_y1 + b2_y2) / 2

                rho2 = (b2_cx - b1_cx)**2 + (b2_cy - b1_cy)**2
                c2 = c_w**2 + c_h**2 + eps

                if CIoU:
                    # aspect ratio consistency
                    w1 = b1_x2 - b1_x1 + eps
                    h1 = b1_y2 - b1_y1 + eps
                    w2 = b2_x2 - b2_x1 + eps
                    h2 = b2_y2 - b2_y1 + eps
                    v = (4/np.pi**2) * (np.arctan(w2/h2) - np.arctan(w1/h1))**2
                    alpha = v / (1 - iou + v + eps)
                    return iou - (rho2 / c2 + v * alpha)
                return iou - rho2 / c2  # DIoU

            c_area = c_w * c_h + eps
            return iou - (c_area - union)/c_area  # GIoU

        return iou


    def reverse_letterbox(self,
                         boxes:np.ndarray,
                         r:float,
                         dwdh:tuple,
                         ):
        """
        将推理阶段缩放 + 填充（letterbox）后的坐标缩放回原图尺度。

        推理前，通常对输入图像做以下预处理：
            1. 按比例缩放：img_resized = img * r
            2. 边缘填充：pad = (dw, dh)

        推理输出的框同样是在 “缩放 + 填充” 后的坐标系中。
        本函数负责将这些框还原回原图坐标系。

        Args:
            boxes (np.ndarray):
                检测框数组，shape 通常为 (N, 4) 或 (B, N, 4)，格式必须为 xyxy。
                其中 x1,y1,x2,y2 均是在 letterbox 后的坐标系中。

            r (float):
                resize 缩放比例：r = new_size / old_size。（letterbox的缩放率）

            dwdh (tuple):
                预处理时的 padding 偏移量 (dw, dh)，即左右/上下的边缘填充像素。

        Returns:
            np.ndarray:
                还原到原图坐标后的框数组，shape 与输入一致。
        """
        dw, dh = dwdh
        boxes = boxes.copy()
        boxes[:, 0, :] = (boxes[:, 0, :] - dw) / r  # x1
        boxes[:, 1, :] = (boxes[:, 1, :] - dh) / r  # y1
        boxes[:, 2, :] = (boxes[:, 2, :] - dw) / r  # x2
        boxes[:, 3, :] = (boxes[:, 3, :] - dh) / r  # y2
        return boxes


    def reverse_scale_pose(self,
                           pose:np.ndarray,
                           r:float,
                           dwdh:tuple):
        """
        将推理阶段缩放 + 填充（letterbox）后的坐标缩放回原图尺度。

        推理前，通常对输入图像做以下预处理：
            1. 按比例缩放：img_resized = img * r
            2. 边缘填充：pad = (dw, dh)

        推理输出的框同样是在 “缩放 + 填充” 后的坐标系中。
        本函数负责将这些框还原回原图坐标系。

        Args:
            pose (np.ndarray):
                人体姿态数组。这里需要是一个形状为 (1, num_keypoints*3, num_boxes) 的数组，2D谷歌点， (x, y, conf)

            r (float):
                resize 缩放比例：r = new_size / old_size。（letterbox的缩放率）

            dwdh (tuple):
                预处理时的 padding 偏移量 (dw, dh)，即左右/上下的边缘填充像素。

        Returns:
            np.ndarray:
                还原到原图坐标后的框数组，shape 与输入一致。
        """
        dw, dh = dwdh
        pose = pose.copy()
        num_point = pose.shape[1] // 3
        num_box = pose.shape[2]
        for i in range(num_box):
            keypoints = pose[0, :, i].reshape(num_point, 3)
            for j in range(num_point):
                x, y, conf = keypoints[j]
                x = (x - dw) / r
                y = (y - dh) / r
                keypoints[j] = [x, y, conf]
            pose[0, :, i] = keypoints.reshape(-1)

        return pose


    def dfl(self,
            position,
            ):
        """
        分布式焦点回归（Distribution Focal Loss, DFL）解码函数。
        将网络输出的边界框分布特征转换为实际边界框偏移量。

        参数:
            position (list | np.ndarray): 网络输出的边界框分布预测，
                                        形状为 (N, C, H, W)
                                        N: 批大小
                                        C: 通道数 (通常为 4 * mc, 4 表示边框四个边)
                                        H: 高度
                                        W: 宽度

        返回:
            np.ndarray: 解码后的边界框偏移量，形状为 (N, 4, H, W)
                        4 对应四个边（left, top, right, bottom）

        处理步骤:
            1. 将输入转换为 numpy 数组，并获取形状 (n, c, h, w)。
            2. 将通道 c 拆分为 p_num (4) 个边框边和每边的 mc 个离散分布维度。
            reshape 成 (n, p_num, mc, h, w)。
            3. 对 mc 维度进行 softmax，得到每个离散位置的概率分布。
            4. 创建累积权重数组 acc = [0, 1, ..., mc-1]，用于计算期望。
            5. 对 mc 维度做加权求和，得到每个边的连续偏移量。
        """
        x = np.array(position)
        n, c, h, w = x.shape
        p_num = 4  # 通常为4个边界框边（left, top, right, bottom）
        mc = c // p_num  # 每个边对应的分布维度

        # reshape: (n, p_num, mc, h, w)
        x = x.reshape(n, p_num, mc, h, w)

        # softmax on mc dimension (axis=2)
        exp_x = np.exp(x - np.max(x, axis=2, keepdims=True))
        softmax_x = exp_x / np.sum(exp_x, axis=2, keepdims=True)

        # acc shape: (1, 1, mc, 1, 1)，用于计算期望
        acc = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)

        # weighted sum over mc axis，得到每个边的连续偏移量
        output = np.sum(softmax_x * acc, axis=2)  # shape: (n, p_num, h, w)

        return output


    def standard_outputs_postprocess(self,
                                     outputs:np.ndarray,
                                     scale_rate:float,
                                     dwdh:tuple[float, float],
                                     pose_model:bool=False,
                                     ) -> Tuple[np.ndarray,
                                                np.ndarray,
                                                np.ndarray,
                                                np.ndarray | None,
                                                ]:
        """
        标准模型输出后处理，适配纯检测模型和姿态模型

        Args:
            outputs (np.ndarray):
                模型输出的原始结果
            scale_rate (float):
                letterbox 缩放率
            dwdh (tuple):
                (dw, dh)，左侧 padding 和上方 padding
            pose_model (bool):
                是否为姿态模型. Defaults to False.

        Returns:
            tuple:
                boxes (np.ndarray):
                    检测框数组，shape=(1, 4, N)，xyxy 格式。
                    N 可以为 0，表示该帧没有检测到目标。
                
                scores (np.ndarray):
                    检测框置信度数组，shape=(1, N)。
                    空数组表示没有检测到目标。

                class_indices (np.ndarray):
                    检测框类别索引数组，shape=(1, N)。
                    空数组表示没有检测到目标。

                pose (np.ndarray | None):
                    姿态关键点数组，shape=(1, nop*dim, N)。
                    pose_flatten为骨骼点数量*骨骼点维度，例如17个点的带置信度的2D骨骼点为51（17 * （2 + 1））。
                    如果 pose_model=False，则返回 None。
                    如果该帧没有检测到目标，则返回 shape=(0, K, D) 的空数组。
        """
        
        if not pose_model:
            boxes = outputs[0][:, :4, :]
            class_indices = np.argmax(outputs[0][:, 4:, :], axis=1)
            scores = np.max(outputs[0][:, 4:, :], axis=1)
        
        else:
            # stand_pose认为只有一个类别
            boxes = outputs[0][:, :4, :]
            class_indices = np.zeros((1, outputs[0].shape[-1]), dtype=np.float32)
            scores = outputs[0][:, 4, :]
            pose = outputs[0][:, 5:, :]

        # 置信度过滤
        mask = scores > self.conf_thres  # shape: (1, 8400)
        mask = mask.reshape(-1)  # shape: (1, 1, 8400)

        # 类别、得分过滤
        class_indices = class_indices[:, mask]  # 保留置信度大于阈值的部分
        scores = scores[:, mask]  # 保留置信度大于阈值的部分
        boxes = boxes[:, :, mask]  # 保留置信度大于阈值的部分
        
        # 调用NMS, 保留NMS后的结果
        nms_keep = self.nms(boxes[0].T, scores[0], class_indices[0], self.nms_thres)  # 传递class_indices
        boxes = boxes[:, :, nms_keep]  # 保留nms过滤后的结果
        class_indices = class_indices[:, nms_keep]  # 保留nms过滤后的结果
        scores = scores[:, nms_keep]  # 保留nms过滤后的结果

        # 尺度恢复
        boxes = self.xywh_to_xyxy(boxes)
        boxes = self.reverse_letterbox(boxes, scale_rate, dwdh)
        
        # pose
        if pose_model:
            pose = pose[:, :, mask] # 保留置信度大于阈值的部分
            pose = pose[:, :, nms_keep] # 保留nms过滤后的结果
            pose = self.reverse_scale_pose(pose, scale_rate, dwdh)
        else:
            pose = None
        
        return boxes, scores, class_indices, pose      

    @abstractmethod
    def __call__(self,
                 outputs:np.ndarray,
                 scale_rate:float,
                 dwdh:tuple[float, float],
                 ) -> Tuple[np.ndarray,
                            np.ndarray,
                            np.ndarray,
                            np.ndarray | None,
                            ]:
        """
        标准模型输出后处理，适配纯检测模型和姿态模型

        Args:
            outputs (np.ndarray):
                模型输出的原始结果
            scale_rate (float):
                letterbox 缩放率
            dwdh (tuple):
                (dw, dh)，左侧 padding 和上方 padding

        Returns:
            tuple:
                boxes (np.ndarray):
                    检测框数组，shape=(1, 4, N)，xyxy 格式。
                    N 可以为 0，表示该帧没有检测到目标。
                
                scores (np.ndarray):
                    检测框置信度数组，shape=(1, N)。
                    N 可以为 0，表示该帧没有检测到目标。

                class_indices (np.ndarray):
                    检测框类别索引数组，shape=(1, N)。
                    N 可以为 0，表示该帧没有检测到目标。

                pose (np.ndarray | None):
                    姿态关键点数组，shape=(1, nop*dim, N)。
                    pose_flatten为骨骼点数量*骨骼点维度，例如17个点的带置信度的2D骨骼点为51（17 * （2 + 1））。
                    如果 pose_model=False，则返回 None。
                    N 可以为 0，表示该帧没有检测到目标。
        """
        pass



class PostProcessor_for_YOLOv5(BasePostProcessor):

    def __init__(self,
                 input_size=(640, 640),
                 conf_thres=0.5,
                 nms_thres=0.5,
                 masks=[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                 anchors=[[10,13], [16,30], [33,23], [30,61], [62,45], [59,119], [116,90], [156,198], [373,326]],
                 ):
        super().__init__(input_size, conf_thres, nms_thres)
        self.masks = masks
        self.anchors = anchors
        self.anchor_points, self.stride_tensor = self.init_anchors(self.feats_hw, self.strides)


    def v5to_v8_style(self, outputs):
        """
        将YOLOv5的输出转化为与YOLOv8一致的格式进行后处理

        Args:
            outputs (list): 模型的输出，后处理之前的结果

        Returns:
            list: 转化后的输出，符合YOLOv8格式
        """
        bbox = outputs[0][..., :4]
        conf = 1 / (1 + np.exp(-outputs[0][..., 4:5]))
        cls  = 1 / (1 + np.exp(-outputs[0][..., 5:]))
        outputs_v8style = [np.concatenate([bbox, cls * conf], axis=-1).transpose(0, 2, 1)]
        return outputs_v8style


    def __call__(self, outputs, scale_rate, dwdh):
        """
        scale_rate: letterbox缩放率
        dwdh: (dw, dh), 左侧padding和上方padding
        """
        if len(outputs) == 1:
            # 标准模型
            outputs_v8style = self.v5to_v8_style(outputs)            
            return self.standard_outputs_postprocess(outputs=outputs_v8style,
                                            scale_rate=scale_rate,
                                            dwdh=dwdh,
                                            pose_model=False,
                                            )
        
        elif len(outputs) == 3:
            # 参考rk: https://github.com/airockchip/rknn_model_zoo/blob/main/examples/yolov5/python/yolov5.py
            def filter_boxes(boxes, box_confidences, box_class_probs):
                """Filter boxes with object threshold.
                """
                box_confidences = box_confidences.reshape(-1)
                class_max_score = np.max(box_class_probs, axis=-1)
                classes = np.argmax(box_class_probs, axis=-1)

                _class_pos = np.where(class_max_score* box_confidences >= self.conf_thres)
                
                scores = (class_max_score* box_confidences)[_class_pos]

                boxes = boxes[_class_pos]

                classes = classes[_class_pos]
                    
                return boxes, classes, scores

            def nms_boxes(boxes, scores):
                """Suppress non-maximal boxes.
                # Returns
                    keep: ndarray, index of effective boxes.
                """
                x = boxes[:, 0]
                y = boxes[:, 1]
                w = boxes[:, 2] - boxes[:, 0]
                h = boxes[:, 3] - boxes[:, 1]

                areas = w * h
                order = scores.argsort()[::-1]

                keep = []
                while order.size > 0:
                    i = order[0]
                    keep.append(i)

                    xx1 = np.maximum(x[i], x[order[1:]])
                    yy1 = np.maximum(y[i], y[order[1:]])
                    xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
                    yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

                    w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
                    h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
                    inter = w1 * h1

                    ovr = inter / (areas[i] + areas[order[1:]] - inter)
                    inds = np.where(ovr <= self.nms_thres)[0]
                    order = order[inds + 1]
                keep = np.array(keep)
                return keep

            def box_process(position, anchors):
                grid_h, grid_w = position.shape[2:4]
                col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
                col = col.reshape(1, 1, grid_h, grid_w)
                row = row.reshape(1, 1, grid_h, grid_w)
                grid = np.concatenate((col, row), axis=1)
                stride = np.array([self.input_size[1]//grid_h, self.input_size[0]//grid_w]).reshape(1,2,1,1)

                col = col.repeat(len(anchors), axis=0)
                row = row.repeat(len(anchors), axis=0)
                anchors = np.array(anchors)
                anchors = anchors.reshape(*anchors.shape, 1, 1)

                box_xy = position[:,:2,:,:]*2 - 0.5
                box_wh = pow(position[:,2:4,:,:]*2, 2) * anchors

                box_xy += grid
                box_xy *= stride
                box = np.concatenate((box_xy, box_wh), axis=1)

                # Convert [c_x, c_y, w, h] to [x1, y1, x2, y2]
                xyxy = np.copy(box)
                xyxy[:, 0, :, :] = box[:, 0, :, :] - box[:, 2, :, :]/ 2  # top left x
                xyxy[:, 1, :, :] = box[:, 1, :, :] - box[:, 3, :, :]/ 2  # top left y
                xyxy[:, 2, :, :] = box[:, 0, :, :] + box[:, 2, :, :]/ 2  # bottom right x
                xyxy[:, 3, :, :] = box[:, 1, :, :] + box[:, 3, :, :]/ 2  # bottom right y

                return xyxy

            def post_process(input_data, anchors):
                boxes, scores, classes_conf = [], [], []
                # 1*255*h*w -> 3*85*h*w
                input_data = [_in.reshape([3, -1] + list(_in.shape[-2:])) for _in in input_data]

                for i in range(len(input_data)):
                    boxes.append(box_process(input_data[i][:,:4,:,:], anchors[i]))
                    scores.append(input_data[i][:,4:5,:,:])
                    classes_conf.append(input_data[i][:,5:,:,:])

                def sp_flatten(_in):
                    ch = _in.shape[1]
                    _in = _in.transpose(0,2,3,1)
                    return _in.reshape(-1, ch)

                boxes = [sp_flatten(_v) for _v in boxes]
                classes_conf = [sp_flatten(_v) for _v in classes_conf]
                scores = [sp_flatten(_v) for _v in scores]
                boxes = np.concatenate(boxes)
                classes_conf = np.concatenate(classes_conf)
                scores = np.concatenate(scores)                
                
                # filter according to threshold
                boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

                # nms
                nboxes, nclasses, nscores = [], [], []

                for c in set(classes):
                    inds = np.where(classes == c)
                    b = boxes[inds]
                    c = classes[inds]
                    s = scores[inds]
                    keep = nms_boxes(b, s)

                    if len(keep) != 0:
                        nboxes.append(b[keep])
                        nclasses.append(c[keep])
                        nscores.append(s[keep])

                if not nclasses and not nscores:
                    return None, None, None

                boxes = np.concatenate(nboxes)
                classes = np.concatenate(nclasses)
                scores = np.concatenate(nscores)

                return boxes, classes, scores

            boxes, classes, scores = post_process(outputs, np.array(self.anchors).reshape(3,-1,2).tolist())
            boxes = np.expand_dims(boxes.T, axis=0)
            boxes = self.reverse_letterbox(boxes, scale_rate, dwdh)
            classes = classes.reshape(1, -1)
            scores = scores.reshape(1, -1)
            if boxes is not None:
                return boxes, scores, classes, None
            else:
                return None, None, None, None

        else:
            raise ValueError(f"模型共{len(outputs)}个输出，暂不支持")



class PostProcessor_for_YOLOv8(BasePostProcessor):
    
    def __init__(self, input_size=(640, 640), conf_thres=0.5, nms_thres=0.5, pose=False):
        super().__init__(input_size, conf_thres, nms_thres)
        self.anchor_points, self.stride_tensor = self.init_anchors(self.feats_hw, self.strides)
        self.pose = pose


    def __call__(self, outputs, scale_rate, dwdh):
        """
        YOLOv8 后处理，适配原版模型和截断模型

        Args:
            outputs (list): 模型的输出，后处理之前的结果
            scale_rate (float): letter box的缩放率
            dwdh (tuple): 左侧padding和上方padding距离

        Returns:
            tuple: boxes, scores, class_indices, None
        """

        if len(outputs) == 1:
            # 标准模型，单个输出
            boxes, scores, class_indices, pose =  self.standard_outputs_postprocess(outputs=outputs,
                                                     scale_rate=scale_rate,
                                                     dwdh=dwdh,
                                                     pose_model=self.pose,
                                                     )
            return boxes, scores, class_indices, pose
            
        elif len(outputs) == 9:
            # 参考自rockhip: https://github.com/airockchip/ultralytics_yolov8/blob/main/RKOPT_README.zh-CN.md
            # https://github.com/airockchip/rknn_model_zoo/tree/main/examples/yolov8

            def filter_boxes(boxes, box_confidences, box_class_probs):
                """Filter boxes with object threshold.
                """
                box_confidences = box_confidences.reshape(-1)

                class_max_score = np.max(box_class_probs, axis=-1)
                classes = np.argmax(box_class_probs, axis=-1)

                _class_pos = np.where(class_max_score* box_confidences >= self.conf_thres)
                scores = (class_max_score* box_confidences)[_class_pos]

                boxes = boxes[_class_pos]
                classes = classes[_class_pos]

                return boxes, classes, scores


            def box_process(position):
                grid_h, grid_w = position.shape[2:4]
                col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
                col = col.reshape(1, 1, grid_h, grid_w)
                row = row.reshape(1, 1, grid_h, grid_w)
                grid = np.concatenate((col, row), axis=1)
                stride = np.array([self.input_size[1]//grid_h, self.input_size[0]//grid_w]).reshape(1,2,1,1)

                position = self.dfl(position)
                box_xy  = grid +0.5 - position[:,0:2,:,:]
                box_xy2 = grid +0.5 + position[:,2:4,:,:]
                xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

                return xyxy


            boxes, scores, classes_conf = [], [], []
            defualt_branch = 3
            pair_per_branch = len(outputs) // defualt_branch
            
            def sp_flatten(_in):
                ch = _in.shape[1]
                _in = _in.transpose(0,2,3,1)
                return _in.reshape(-1, ch)
            
            
            for i in range(defualt_branch): # Python 忽略 score_sum 输出 9个只取6个
                boxes.append(sp_flatten(box_process(outputs[pair_per_branch*i]))) # box_process: xyxy
                classes_conf.append(sp_flatten(outputs[pair_per_branch*i+1]))
                scores.append(sp_flatten(np.ones_like(outputs[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32)))


            if len(boxes) > 0:
                boxes = np.concatenate(boxes)
                classes_conf = np.concatenate(classes_conf)
                scores = np.concatenate(scores)

                # filter according to threshold

                boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
                nms_keep = self.nms(boxes=boxes, scores=scores, class_ids=classes, iou_threshold=self.nms_thres, xywh=False)
                
                boxes = np.expand_dims(boxes.T, axis=0)
                class_indices = np.expand_dims(classes, axis=0)
                scores = np.expand_dims(scores, axis=0)
                
                boxes = boxes[:, :, nms_keep]  # 保留nms过滤后的结果
                class_indices = class_indices[:, nms_keep]  # 保留nms过滤后的结果
                scores = scores[:, nms_keep]  # 保留nms过滤后的结果
                boxes = self.reverse_letterbox(boxes, scale_rate, dwdh)
        
        else:
            raise ValueError(f"模型共{len(outputs)}个输出，暂不支持")

        return boxes, scores, class_indices, pose



class PostProcessor_for_RopeSkipping(BasePostProcessor):
    
    def __init__(self, input_size=(640, 640), conf_thres=0.5, nms_thres=0.5):
        super().__init__(input_size, conf_thres, nms_thres)
        self.anchor_points, self.stride_tensor = self.init_anchors(self.feats_hw, self.strides)
        print("\n\033[32m初始化跳绳后处理, 继承后处理基类(postprocessor.BasePostProcessor), 包含操作:\n被截断的模型部分(三输出到最终单输出) -> 置信度过滤 -> NMS过滤(针对跳绳优化的NMS, 不针对每个类单独进行) -> 尺度恢复\n\033[0m")


    def decode_bboxes(self, dfl_output):
        lt, rb = np.array_split(dfl_output, 2, axis=1)
        x1y1 = self.anchor_points - lt
        x2y2 = self.anchor_points + rb
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate([c_xy, wh], axis=1) * self.stride_tensor


    def nms_xywh(self, boxes, scores, iou_threshold=0.5):
        """
        针对跳绳项目修改, 跨类别NMS
        """
        boxes_xyxy = self.xywh_to_xyxy(boxes)
        indices = scores.argsort()[::-1]
        keep = []

        while indices.size > 0:
            current = indices[0]
            keep.append(current)

            if indices.size == 1:
                break

            rest_boxes = boxes_xyxy[indices[1:]]
            ious = self.compute_iou(boxes_xyxy[current], rest_boxes)

            indices = indices[1:][ious < iou_threshold]

        return keep


    def __call__(self, outputs, scale_rate, dwdh):
        """
        scale_rate: letter box的缩放率
        dwdh: (dw, dh), 左侧padding和上方padding距离
        """

        # 类别、得分
        cls_80 = self.sigmoid(outputs[0][:, 64:, :, :])
        cls_40 = self.sigmoid(outputs[1][:, 64:, :, :])
        cls_20 = self.sigmoid(outputs[2][:, 64:, :, :])
        cls = np.concatenate(
                            [
                            cls_80.reshape(1, 6, -1), 
                            cls_40.reshape(1, 6, -1), 
                            cls_20.reshape(1, 6, -1)
                            ], 
                            axis=2) # cls:  (1, 6, 8400)

        class_indices = np.argmax(cls, axis=1)  # shape: (1, 8400)
        scores = np.max(cls, axis=1) # shape: (1, 8400)
        mask = scores > self.conf_thres  # shape: (1, 8400)
        mask = mask.reshape(-1) # shape: (1, 1, 8400)
        # 类别、得分过滤
        class_indices = class_indices[:, mask] # 保留置信度大于阈值的部分
        cls = cls[:, :, mask] # 保留置信度大于阈值的部分
        scores = scores[:, mask] # 保留置信度大于阈值的部分

        # 检测框和类别
        box_80 = self.dfl(outputs[0][:, :64, :, :]).reshape(1, 4, -1)
        box_40 = self.dfl(outputs[1][:, :64, :, :]).reshape(1, 4, -1)
        box_20 = self.dfl(outputs[2][:, :64, :, :]).reshape(1, 4, -1)
        box = np.concatenate([box_80, box_40, box_20], axis=2)
        boxes = self.decode_bboxes(box) # (1, 4, 8400) # 这里boxes对了，和源代码的decoder_bboxes输出一样！！！！！
        # 检测框和类别过滤
        boxes =  boxes[:, :, mask] # 保留置信度大于阈值的部分
        nms_keep = self.nms_xywh(boxes[0].T, scores[0], self.nms_thres)
        boxes =  boxes[:, :, nms_keep] # 保留nms过滤后的结果
        class_indices = class_indices[:, nms_keep] # 保留置信度大于阈值的部分
        cls = cls[:, :, nms_keep] # 保留nms过滤后的结果
        scores = scores[:, nms_keep] # 保留nms过滤后的结果

        # 骨骼点
        pose = outputs[3].reshape(1, 105, -1) # pose:  (1, 105, 8400)
        pose = pose[:, :, mask] # 保留置信度大于阈值的部分
        pose = pose[:, :, nms_keep] # 保留nms过滤后的结果
        
        # 尺度恢复
        boxes = self.xywh_to_xyxy(boxes)
        boxes = self.reverse_letterbox(boxes, scale_rate, dwdh)
        pose = self.reverse_scale_pose(pose, scale_rate, dwdh)
        
        # (batch_size, 4, box个数) (batch_size, box个数) (batch_size, box个数) (batch_size, 骨骼点数量*3, box个数)
        return boxes, scores, class_indices, pose



class PostProcessor_for_YOLOv8_Face(BasePostProcessor):
    def __init__(self, input_size=(640, 640), conf_thres=0.5, nms_thres=0.5):
        super().__init__(input_size, conf_thres, nms_thres)
        self.reg_max = 16
        self.project = np.arange(self.reg_max)
        self.anchor_points, self.stride_tensor = self.init_anchors(self.feats_hw, self.strides)
        print("\n\033[32m初始化YOLOFace后处理, 继承后处理基类(postprocessor.BasePostProcessor), 包含操作:\n被截断的模型部分(三输出到最终单输出) -> 置信度过滤 -> NMS过滤(不针对每个类单独进行) -> 尺度恢复\n\033[0m")


    def reconst_xywh_to_xyxy(self, boxes):
        """Convert [x_lu, y_lu, w, h] -> [x1, y1, x2, y2] 很离谱，这个模型的xywh的xy是左上角而不是中点"""
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x2 = x + w
        y2 = y + h
        return np.stack([x, y, x2, y2], axis=1)


    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s


    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    
    def __call__(self, outputs, scale_rate, dwdh):
        if len(outputs) == 1:
            # 标准模型，单个输出
            boxes = outputs[0][:, :4, :]  # boxes.shape: (1, 4, 8400)
            scores = outputs[0][:, 4, :]
            pose = outputs[0][:, 5:, :]  # landmarks.shape: (1, 15, 8400)
            class_indices = np.zeros_like(scores, dtype=np.int32)  # shape: (1, 8400)
            
            # 置信度过滤
            mask = scores > self.conf_thres  # shape: (1, 8400)
            mask = mask.reshape(-1)  # shape: (1, 1, 8400)
        
            # 类别、得分过滤
            class_indices = class_indices[:, mask]  # 保留置信度大于阈值的部分
            scores = scores[:, mask]  # 保留置信度大于阈值的部分
            boxes = boxes[:, :, mask]  # 保留置信度大于阈值的部分

            # 调用NMS, 保留NMS后的结果
            nms_keep = self.nms(boxes[0].T, scores[0], class_indices[0], self.nms_thres)  # 传递class_indices
            boxes = boxes[:, :, nms_keep]  # 保留nms过滤后的结果
            class_indices = class_indices[:, nms_keep]  # 保留nms过滤后的结果
            scores = scores[:, nms_keep]  # 保留nms过滤后的结果

            # 尺度恢复
            boxes = self.xywh_to_xyxy(boxes)
            boxes = self.reverse_letterbox(boxes, scale_rate, dwdh)
        
            # 姿态
            pose = pose[:, :, mask] # 保留置信度大于阈值的部分
            pose = pose[:, :, nms_keep] # 保留nms过滤后的结果
            pose = self.reverse_scale_pose(pose, scale_rate, dwdh)
        
            return boxes, scores, class_indices, pose
        
        if len(outputs) == 3:
            # 三输出的截断模型，参考自 https://github.com/derronqi/yolov8-face
            bboxes, scores, landmarks = [], [], []
            for i, pred in enumerate(outputs):
                stride = int(self.input_size[0] / pred.shape[2])
                pred = pred.transpose((0, 2, 3, 1))
                box = pred[..., :self.reg_max * 4]
                cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1,1))
                kpts = pred[..., -15:].reshape((-1,15)) ### x1,y1,score1, ..., x5,y5,score5
                tmp = box.reshape(-1, 4, self.reg_max)
                bbox_pred = self.softmax(tmp, axis=-1)
                bbox_pred = np.dot(bbox_pred, self.project).reshape((-1,4))
                bbox = self.distance2bbox(self.anchor_points[i], bbox_pred, max_shape=self.input_size) * stride

                kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchor_points[i][:, 0].reshape((-1,1)) - 0.5)) * stride
                kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchor_points[i][:, 1].reshape((-1,1)) - 0.5)) * stride
                kpts[:, 2::3] = 1 / (1+np.exp(-kpts[:, 2::3]))

                bbox -= np.array([[dwdh[0], dwdh[1], dwdh[0], dwdh[1]]])  ###合理使用广播法则
                bbox /= np.array([[scale_rate, scale_rate, scale_rate, scale_rate]])

                kpts -= np.tile(np.array([dwdh[0], dwdh[1], 0]), 5).reshape((1,15))
                kpts /= np.tile(np.array([scale_rate, scale_rate, 1]), 5).reshape((1,15))

                bboxes.append(bbox)
                scores.append(cls)
                landmarks.append(kpts)
            bboxes = np.concatenate(bboxes, axis=0)
            scores = np.concatenate(scores, axis=0)
            landmarks = np.concatenate(landmarks, axis=0)
        
            bboxes_wh = bboxes.copy()

            bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  ####xywh
            classIds = np.argmax(scores, axis=1)
            confidences = np.max(scores, axis=1)  ####max_class_confidence
            
            mask = confidences > self.conf_thres
            bboxes_wh = bboxes_wh[mask]  ###合理使用广播法则
            confidences = confidences[mask]
            classIds = classIds[mask]
            landmarks = landmarks[mask]
            indices = self.nms(bboxes_wh, confidences, classIds, iou_threshold=self.nms_thres, xywh=True)
            mlvl_bboxes = np.expand_dims(bboxes_wh[indices].T, axis=0)
            confidences = np.expand_dims(confidences[indices].T, axis=0)
            classIds = np.expand_dims(classIds[indices], axis=0)
            landmarks = np.expand_dims(landmarks[indices].T, axis=0)
            
            if len(indices) > 0:
                mlvl_bboxes = self.reconst_xywh_to_xyxy(mlvl_bboxes)
                # (batch_size, 4, box个数) (batch_size, box个数) (batch_size, box个数) (batch_size, 骨骼点数量*3, box个数)
                return mlvl_bboxes, confidences, classIds, landmarks
            else:
                return np.array([]), np.array([]), np.array([]), np.array([])


class PostProcessor_for_YOLOv26(BasePostProcessor):
    """适配 YOLOv26 模型的后处理类"""

    def __init__(self,
                 input_size: Tuple[int, int] = (640, 640),
                 conf_thres: float = 0.5,
                 nms_thres: float = 0.5):
        """初始化 YOLOv26 后处理类

        Args:
            input_size (Tuple[int, int]): 模型输入图像的尺寸 (height, width)。
            conf_thres (float): 置信度阈值。
            nms_thres (float): 非极大值抑制 (NMS) 的 IoU 阈值。
        """
        super().__init__(input_size, conf_thres, nms_thres)

        # 利用基类 init_anchors 预先生成全量锚点和步长张量
        self.anchor_points, self.stride_tensor = self.init_anchors(self.feats_hw, self.strides)

    def _process_end2end_1_head(self,
                                outputs: List[np.ndarray],
                                scale_rate: float,
                                dwdh: Tuple[float, float]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        处理端到端 (End-to-End) 模型输出，无需 NMS。通常输出形状为 [1, 300, 6]。
        6个维度的含义为 [x1, y1, x2, y2, score, class_id]
        """
        # 取出 batch 中的第一个数据，形状变为 (300, 6)
        data = outputs[0][0]

        # 1. 直接拆分出 坐标、置信度 与 类别
        boxes_raw = data[:, :4]  # 已经是 xyxy 格式 (300, 4)
        scores = data[:, 4]  # 置信度 (300,)
        class_ids = data[:, 5].astype(np.int32)  # 类别 (300,)

        # 2. 基于置信度阈值过滤无效框
        mask = scores > self.conf_thres

        valid_boxes = boxes_raw[mask]
        valid_scores = scores[mask]
        valid_class_ids = class_ids[mask]

        # 如果没有符合条件的框，返回空数组
        if len(valid_boxes) == 0:
            return np.empty((1, 4, 0)), np.empty((1, 0)), np.empty((1, 0)), None

        # 3. 维度调整以匹配后续程序的期望格式
        # 你的主程序期望 boxes: (1, 4, N), scores: (1, N), class_ids: (1, N)
        final_boxes = np.expand_dims(valid_boxes.T, axis=0)  # 转置并增加 batch 维度
        final_scores = np.expand_dims(valid_scores, axis=0)
        final_class_ids = np.expand_dims(valid_class_ids, axis=0)

        # 4. 尺度恢复回原图大小
        # 注意：这里不需要再调用 self.xywh_to_xyxy，因为模型直接输出的就是 xyxy
        final_boxes = self.reverse_letterbox(final_boxes, scale_rate, dwdh)

        return final_boxes, final_scores, final_class_ids, None

    def _process_standard_1_head(self,
                                 outputs: List[np.ndarray],
                                 scale_rate: float,
                                 dwdh: Tuple[float, float]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """处理标准的单输出头，通常输出形状为 [1, 84, 8400]"""
        data = outputs[0]  # shape: (1, 84, 8400)

        # 1. 拆分坐标与分类信息
        boxes_raw = data[:, :4, :]
        # 直接读取分类概率 (已在导出模型时激活，切勿进行二次 Sigmoid)
        cls_scores = data[:, 4:, :]

        # 2. 计算最大得分与对应类别
        scores = np.max(cls_scores, axis=1)
        class_ids = np.argmax(cls_scores, axis=1)

        # 3. 基于置信度阈值进行初步过滤
        mask = scores[0] > self.conf_thres
        if not np.any(mask):
            return np.empty((1, 4, 0)), np.empty((1, 0)), np.empty((1, 0)), None

        valid_boxes = boxes_raw[:, :, mask]
        valid_scores = scores[:, mask]
        valid_class_ids = class_ids[:, mask]

        # 4. 执行非极大值抑制 (NMS)
        keep = self.nms(
            boxes=valid_boxes[0].T,
            scores=valid_scores[0],
            class_ids=valid_class_ids[0],
            iou_threshold=self.nms_thres,
            xywh=True
        )

        if len(keep) == 0:
            return np.empty((1, 4, 0)), np.empty((1, 0)), np.empty((1, 0)), None

        final_boxes = valid_boxes[:, :, keep]
        final_scores = valid_scores[:, keep]
        final_class_ids = valid_class_ids[:, keep]

        final_boxes = self.xywh_to_xyxy(final_boxes)
        final_boxes = self.reverse_letterbox(final_boxes, scale_rate, dwdh)

        return final_boxes, final_scores, final_class_ids, None

    def _process_rknn_6_heads(self,
                              outputs: List[np.ndarray],
                              scale_rate: float,
                              dwdh: Tuple[float, float]) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """处理 RKNN 常见的 6 头输出 (3个回归头 + 3个分类头)"""
        regs, clss = [], []

        for out in outputs:
            out = out.reshape(out.shape[0], out.shape[1], -1)
            if out.shape[1] == 4:
                regs.append(out)
            else:
                clss.append(out)

        regs = sorted(regs, key=lambda x: x.shape[2], reverse=True)
        clss = sorted(clss, key=lambda x: x.shape[2], reverse=True)

        final_regs = np.concatenate(regs, axis=2)
        final_clss = self.sigmoid(np.concatenate(clss, axis=2))

        lt = final_regs[:, :2, :]
        rb = final_regs[:, 2:, :]

        x1y1 = (self.anchor_points - lt) * self.stride_tensor
        x2y2 = (self.anchor_points + rb) * self.stride_tensor
        boxes = np.concatenate([x1y1, x2y2], axis=1)

        scores = np.max(final_clss, axis=1)
        class_ids = np.argmax(final_clss, axis=1)

        mask = scores[0] > self.conf_thres
        if not np.any(mask):
            return np.empty((1, 4, 0)), np.empty((1, 0)), np.empty((1, 0)), None

        valid_boxes = boxes[:, :, mask]
        valid_scores = scores[:, mask]
        valid_class_ids = class_ids[:, mask]

        keep = self.nms(
            boxes=valid_boxes[0].T,
            scores=valid_scores[0],
            class_ids=valid_class_ids[0],
            iou_threshold=self.nms_thres,
            xywh=False
        )

        if len(keep) == 0:
            return np.empty((1, 4, 0)), np.empty((1, 0)), np.empty((1, 0)), None

        final_boxes = valid_boxes[:, :, keep]
        final_scores = valid_scores[:, keep]
        final_class_ids = valid_class_ids[:, keep]
        final_boxes = self.reverse_letterbox(final_boxes, scale_rate, dwdh)

        return final_boxes, final_scores, final_class_ids, None

    def __call__(self,
                 outputs: Union[np.ndarray, List[np.ndarray]],
                 scale_rate: float,
                 dwdh: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """执行后处理逻辑入口"""
        # 统一输入格式为列表
        process_input = [outputs] if isinstance(outputs, np.ndarray) else outputs

        # 判断是否为 [1, 300, 6] 的端到端 (NMS-Free) 输出
        if len(process_input) == 1 and process_input[0].shape[-1] == 6:
            return self._process_end2end_1_head(process_input, scale_rate, dwdh)

        # 传统的单输出头 (如 [1, 84, 8400])
        elif len(process_input) == 1:
            return self._process_standard_1_head(process_input, scale_rate, dwdh)

        # 6个输出头：保持原有的 RKNN 处理逻辑
        elif len(process_input) == 6:
            return self._process_rknn_6_heads(process_input, scale_rate, dwdh)

        else:
            raise ValueError(f"YOLOv26 暂不支持输出数量为 {len(process_input)} 且格式未知的模型适配")