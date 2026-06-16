import os
import cv2
import numpy as np
from abc import ABC, abstractmethod

class BasePreProcessor(ABC):
    def __init__(self, input_size=(640, 640)):
        self.input_size = input_size

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
        shape = im.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        # r: 缩放率, (dw, dh): 左侧padding和上方padding距离
        return im, r, (dw, dh)

    @abstractmethod
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            pass
        elif isinstance(img, str):
            if not os.path.exists(img):
                raise FileNotFoundError(f"Image not found: {img}")
            img = cv2.imread(img)
        else:
            print("前处理输入的图片为路径或cv2图像")

        src_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 图像的输入确保在RGB格式上
        img, scale_rate, dwdh = self.letterbox(img, self.input_size)

        return src_img, np.expand_dims(img, axis=0), scale_rate, dwdh


class PreProcessor_for_YOLO_general(BasePreProcessor):
    def __init__(self, input_size):
        super().__init__(input_size)
        print("\n\033[32m初始化通用前处理模块, 完全继承前处理基类(preprocessor.BasePreProcessor), 包含操作:\ncv2读取图像 -> 转为RGB -> letterbox -> 返回: (原图, 模型输入, letterbox缩放尺度, letterbox填充边界)\n\033[0m")

    def __call__(self, image_path):
        return super().__call__(image_path)