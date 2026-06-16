import cv2
import numpy as np
import os

class Visualizer:
    def __init__(self, class_names=None, show_boxes=True, show_points=True):
        if class_names is None:
            self.class_names = [str(i) for i in range(80)]
        else:
            self.class_names = class_names
        self.colors = self._generate_colors(len(class_names) if class_names else 80)
        self.show_boxes = show_boxes
        self.show_points = show_points

    def _generate_colors(self, num_classes):
        np.random.seed(42)
        return [tuple(np.random.randint(0, 256, 3).tolist()) for _ in range(num_classes)]

    def visualize(self,
                  img:np.ndarray,
                  boxes:np.ndarray,
                  scores:np.ndarray,
                  cls:np.ndarray,
                  pose:np.ndarray|None=None,
                  track_id:np.ndarray|None=None,
                  save_path:str|None=None,
                  ):
        """可视化

        Args:
            img (np.ndarray): 输入图片
            boxes (np.ndarray): (1, 4, n)的形状
            scores (np.ndarray): (1, n)的形状
            pose (np.ndarray|None, optional): 2D姿态，（1, 51, n)的形状
            track_id (np.ndarray|None, optional): 跟踪ID. Defaults to None.
            save_path (str|None, optional): 保存路径. Defaults to None.

        Returns:
            _type_: _description_
        """
        if isinstance(img, str):
            img = cv2.imread(img)

        if boxes is None or len(boxes) == 0:
            return img

        boxes = boxes.squeeze(0)  # (4, N)
        
        cls = cls.squeeze(0)      # (N,)
        
        if pose is not None:
            pose = pose.squeeze(0)    # (105, N)

        if track_id is not None:
            track_id = track_id.squeeze(0)  # (N,)
        
        img = img.copy()
        N = boxes.shape[-1]


        for i in range(N):
            x1, y1, x2, y2 = map(int, boxes[:, i])
            class_id = int(cls[i])
            color = self.colors[class_id % len(self.colors)]  # 防止越界

            if self.show_boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)
                label = self.class_names[class_id] if self.class_names and class_id < len(self.class_names) else str(class_id)
                label += f", {scores[0, i]:.2f}"
                label += f", ID:{track_id[i]}" if track_id is not None else ""
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.0005 * img.shape[1], color, 2)


            if pose is not None and self.show_points:
                keypoints = pose[:, i].reshape(-1, 3)
                for x, y, conf in keypoints:
                    if conf > 0:
                        cv2.circle(img, (int(x), int(y)), radius=int(0.00234 * img.shape[1]), color=color, thickness=-1) # 0.00234 = 1.5 / 640

        if save_path:
            cv2.imwrite(save_path, img)
            print(f"图片已保存到: {save_path}")

        return img
