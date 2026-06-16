from .preprocessor import *
from .inferencer import Inferencer
from .postprocessor import *
from .visualizer import Visualizer
from .tracker.byte_tracker import BYTETracker, btrack
import os, cv2, re, time
import yaml
from datetime import datetime
import queue
from concurrent.futures import ThreadPoolExecutor


class BasePipeline:
    def __init__(self,
                 model_path:str,
                 input_size:tuple=(640, 640),
                 conf_thres:float=0.5,
                 nms_thres:float=0.5,
                 class_names=None,
                 max_queue_size:int=50,
                 model_parallel:int=12,
                 pose:bool=False,
                 show:bool=False,
                 use_track:bool=False,
                 track_thresh:float=0.5,
                 track_buffer:int=30,
                 match_thresh:float=0.8,
                 tracked_cls_idx:int=0,
                 mot20:bool=False,
                 ):
        """初始化配置

        Args:
            model_path (str): 模型路径
            input_size (tuple, optional): 输入尺寸. Defaults to (640, 640).
            conf_thres (float, optional): 置信度阈值. Defaults to 0.5.
            nms_thres (float, optional): NMS阈值. Defaults to 0.5.
            class_names (list | None, optional): 类别名字. Defaults to None.
            max_queue_size (int, optional): 异步推理视频文件的队列长度. Defaults to 50.
            model_parallel (int, optional): 模型并行推理的模型数. Defaults to 12.
            show (bool, optional): 推理视频时当有GUI界面进行实时展示. Defaults to False.
            use_track (bool, optional): 是否使用多目标跟踪. Defaults to False.
            track_thresh (float, optional): 跟踪器置信度阈值. Defaults to 0.5.
            track_buffer (int, optional): 跟踪器最大丢失帧数. Defaults to 30.
            match_thresh (float, optional): 跟踪器匹配阈值. Defaults to 0.8.
            tracked_cls_idx (int, optional): 跟踪器跟踪的类别索引. Defaults to 0.
            mot20 (bool, optional): 是否使用MOT20的跟踪参数. Defaults to False.
        """
        self.model_path = model_path
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.class_names = class_names
        self.max_queue_size = max_queue_size
        self.model_parallel = model_parallel
        self.pose = pose
        self.show = show
        self.use_track = use_track
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20
        self.init_blocks()
        
        
    def init_blocks(self):
        """
        对于不同的前处理、后处理、可视化、跟踪等模块, 子类可重写该方法进行初始化
        """
        self.inference = Inferencer(self.model_path)
        self.preprocess = PreProcessor_for_YOLO_general(self.input_size)
        self.postprocess = BasePostProcessor(self.input_size, self.conf_thres, self.nms_thres)
        self.visualizer = Visualizer(self.class_names)
        self.tracker = BYTETracker(self.track_thresh,
                                    self.track_buffer,
                                    self.match_thresh,
                                    self.mot20) if self.use_track else None


    def __enter__(self):
        self.inference.load_model()
        self.inference.init_runtime()
        print("\n\033[32m加载模型并初始化\033[0m\n")
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.inference.release()
        print("\n\033[32m释放模型\033[0m\n")


    def _run_image(self, image_path, save_path=None):
        """
        private方法, 该方法不会释放模型 慎重使用
        """
        t0 = time.time()
        src_img, img_tensor, scale_rate, dwdh = self.preprocess(image_path)
        t1 = time.time()
        outputs = self.inference.infer(img_tensor)
        t2 = time.time()
        boxes, scores, cls, pose = self.postprocess(outputs, scale_rate, dwdh)
        t3 = time.time()
        print(f"YOLO  -  Pre: {(t1 - t0)*1e3:.1f}ms | Infer: {(t2 - t1)*1e3:.1f}ms | Post: {(t3 - t2)*1e3:.1f}ms | Total: {(t3 - t0)*1e3:.1f}ms")
        self.plugin(boxes, scores, cls, pose, src_img, image_path)
        if save_path:
            self.visualizer.visualize(src_img, boxes, scores, cls, pose, save_path=save_path)
        return boxes, scores, cls, pose, src_img


    def _run_video(self, video_path, save_path=None):
        """
        该方法不会释放模型. 当返回None时代表流正常获取，返回1时代表流终止
        """
        if video_path == "0":
            video_path = 0
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if save_path:
            out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        frame_idx = 0
        track_id = None
        start = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                end = time.time()
                print(f"视频全程推理时间: {(end - start):.2f}s, 平均FPS: {int(frame_idx // (end - start))}")
                print(f"视频保存到: {save_path}")
                return 1 # 返回1时代表流终止
            t0 = time.time()
            src_img, img_tensor, scale_rate, dwdh = self.preprocess(frame)
            t1 = time.time() # t0~t1: 前处理
            outputs = self.inference.infer(img_tensor)
            t2 = time.time() # t1~t2: 推理
            boxes, scores, cls, pose = self.postprocess(outputs, scale_rate, dwdh)

            t3 = time.time() # t2~t3: 后处理
            # print("原本box:")
            # print("boxes: ", boxes, type(boxes), boxes.shape)
            # print("scores: ", scores)
            if self.use_track and not self.pose and boxes.shape[-1] > 0: # 这个bytetrack会打乱box顺序，暂不支持yolopose跟踪，需要后续增加索引匹配
                online_tlwhs, online_ids, online_scores = btrack(self.tracker, boxes, scores)
                boxes = np.expand_dims(np.array(online_tlwhs).T, axis=0)
                # print("boxes: ", boxes.shape) # 1, 4, N
                boxes[:, 2, :] += boxes[:, 0, :]
                boxes[:, 3, :] += boxes[:, 1, :]
                track_id = np.expand_dims(np.array(online_ids), axis=0)
                scores = np.expand_dims(np.array(online_scores), axis=0)
                # print("-"* 20)
                # print("跟踪结果")
                # print("boxes: ", boxes)
                # print("track_id: ", track_id)
                # print("scores: ", scores)
                # print("-"* 20)
                
            t4 = time.time() # t3~t4: 跟踪
            self.plugin(boxes, scores, cls, pose, src_img, video_path)
            t5 = time.time() # t4~t5: 插入模块
            if save_path:
                vis_frame = self.visualizer.visualize(src_img, boxes, scores, cls, pose, track_id, None)
                out.write(vis_frame)
                t6 = time.time()
            else:
                t6 = t5
            print(
                f"YOLO - frame_idx: {frame_idx:5d} - "
                f"Pre: {(t1 - t0)*1e3:.1f}ms | "
                f"Infer: {(t2 - t1)*1e3:.1f}ms | "
                f"Post: {(t3 - t2)*1e3:.1f}ms | "
                f"Tracker: {(t4 - t3)*1e3:.1f}ms | "
                f"Plugin: {(t5 - t4)*1e3:.1f}ms | "
                f"Writer: {(t6 - t5)*1e3:.1f}ms | "
                f"Total: {(t6 - t0)*1e3:.1f}ms"
            )

            frame_idx += 1


    def _run_video_async(self, video_path, save_path=None, max_queue_size=50, model_parallel=12, show=False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) if video_path != 0 else 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print(f"FPS: {fps}, w * h = {width} * {height}")
        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height)) if save_path else None

        frame_queue = queue.Queue(maxsize=max_queue_size)
        result_queue = queue.Queue(maxsize=max_queue_size)
        display_queue = queue.Queue(maxsize=max_queue_size)

        next_frame_idx = 0
        result_buffer = {}

        # 初始化多个模型实例
        model_instances = [Inferencer(self.inference.model_path, i % 3) for i in range(model_parallel)]
        for model in model_instances:
            model.load_model()
            model.init_runtime()

        # ---------- 读帧线程 ----------
        def read_frames():
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    for _ in range(model_parallel):
                        frame_queue.put((None, None))
                    break
                frame_queue.put((idx, frame))
                idx += 1

        # ---------- 推理线程 ----------
        def infer_frames(model_idx, model_instance):
            while True:
                idx, frame = frame_queue.get()
                if idx is None:
                    result_queue.put((None, None))
                    break
                t0 = time.time()
                src_img, img_tensor, scale, dwdh = self.preprocess(frame)
                t1 = time.time()
                outputs = model_instance.infer(img_tensor)
                t2 = time.time()
                boxes, scores, cls, pose = self.postprocess(outputs, scale, dwdh)
                t3 = time.time()
                print(f"[模型{model_idx}推理线程] YOLO  - frame: {next_frame_idx} - Pre: {(t1 - t0)*1e3:.1f}ms | Infer: {(t2 - t1)*1e3:.1f}ms | Post: {(t3 - t2)*1e3:.1f}ms | Total: {(t3 - t0)*1e3:.1f}ms ｜ result_queue.size={result_queue.qsize()}")

                if self.use_track and not self.pose and boxes.shape[-1] > 0: # 这个bytetrack会打乱box顺序，暂不支持yolopose跟踪，需要后续增加索引匹配
                    online_tlwhs, online_ids, online_scores = btrack(self.tracker, boxes, scores)
                    boxes = np.expand_dims(np.array(online_tlwhs).T, axis=0)
                    # print("boxes: ", boxes.shape) # 1, 4, N
                    boxes[:, 2, :] += boxes[:, 0, :]
                    boxes[:, 3, :] += boxes[:, 1, :]
                    track_id = np.expand_dims(np.array(online_ids), axis=0)
                    scores = np.expand_dims(np.array(online_scores), axis=0)
                    # print("-"* 20)
                    # print("跟踪结果")
                    # print("boxes: ", boxes)
                    # print("track_id: ", track_id)
                    # print("scores: ", scores)
                    # print("-"* 20)


                result_queue.put((idx, (boxes, scores, cls, pose, src_img)))

        # ---------- 写视频（可视化）线程 ----------
        def visualize_frames():
            nonlocal next_frame_idx
            end_signals = 0

            while True:
                idx, result = result_queue.get()

                if result is None:
                    end_signals += 1
                    if end_signals >= model_parallel:
                        break
                    continue

                result_buffer[idx] = result

                # 按顺序写出
                while next_frame_idx in result_buffer:
                    boxes, scores, cls, pose, src_img = result_buffer.pop(next_frame_idx)
                    self.plugin(boxes, scores, cls, pose, src_img, video_path)
                    vis = self.visualizer.visualize(src_img, boxes, scores, cls, pose, None)

                    if out:
                        out.write(vis)
                    if show:
                        if not display_queue.full():
                            display_queue.put(vis)

                    next_frame_idx += 1

        # ---------- 启动后台线程 ----------
        executor = ThreadPoolExecutor(max_workers=model_parallel + 2)
        futures = []
        futures.append(executor.submit(read_frames))
        for i, m in enumerate(model_instances):
            futures.append(executor.submit(infer_frames, i, m))
        futures.append(executor.submit(visualize_frames))

        try:
            # ---------- 主线程实时显示 ----------
            if show:
                end = False
                while not end:
                    try:
                        frame = display_queue.get(timeout=0.04)
                        cv2.imshow("Realtime", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    except queue.Empty:
                        pass

                    end = all(f.done() for f in futures)

                cv2.destroyAllWindows()

            # 等待线程结束
            for f in futures:
                f.result()

        except KeyboardInterrupt:
            print("\n[Ctrl+C] 用户手动中断，开始安全退出...")
            # 不再往队列写数据，让线程自然退出
            # 此处不 kill 线程，让线程自动完成已经在处理的帧

        finally:
            print("正在释放资源 ...")
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            print(f"资源释放完成。视频文件已安全保存: {save_path}")

        for m in model_instances:
            m.release()


    def plugin(self, boxes, scores, cls, pose, src_img, image_path):
        """
        子类通过重写plugin可增加视频条件处理
        """
        pass


    def run_image(self, image_path, save_path=None):
        """
        这里用with控制上下文, 使用__exit__自动释放模型; 若需要loop调用不要用这个函数, 用 `_run_image`
        """
        with self:
            return self._run_image(image_path, save_path)


    def run_video(self, video_path, save_path):
        with self:
            if self.model_parallel > 1:
                return self._run_video_async(video_path, save_path, self.max_queue_size, self.model_parallel, self.show)
            else:
                return self._run_video(video_path, save_path) 
            


    def run_file(self, file_path, save_dir):
        """
        通用运行模块, 自动识别文件类型并推理
        """
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            self._run_image(file_path, save_dir)
        elif file_path.lower().endswith(('.mp4', '.avi')):
            self.run_video(file_path, save_dir)
        else:
            raise ValueError("不支持的文件类型")


    def run(self, file_path, save_dir=None):
        """
        通用运行模块, 自动识别文件夹或文件, 最高级自动化
        """
        with self:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            
            if os.path.isfile(file_path):
                if save_dir:
                    save_dir = os.path.join(save_dir, os.path.basename(file_path))
                self.run_file(file_path, save_dir)

            elif os.path.isdir(file_path):
                for file in sorted(os.listdir(file_path)):
                    if save_dir:
                        save_path = os.path.join(save_dir, os.path.basename(file))
                    else:
                        save_path = None
                    self.run_file(os.path.join(file_path, file), save_path)

            elif file_path == 0 or file_path == "0":
                # local cam
                if not os.path.exists(file_path):
                    file_path = 0
                    print("本地cam")
                    if save_dir:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        save_dir = os.path.join(save_dir, f"{timestamp}.mp4")
                    self.run_video(file_path, save_dir)
                else:
                    raise ValueError("参数source设置为0时使用本地摄像头作为源，但存在名为0的文件夹")

            elif bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", file_path)):
                # web cam 这里file_path是ip
                if save_dir:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    save_dir = os.path.join(save_dir, f"{timestamp}.mp4")
                username = os.environ.get('RTSP_USERNAME', 'admin')
                password = os.environ.get('RTSP_PASSWORD', '')
                video_url = f"rtsp://{username}:{password}@{file_path}/video"
                self.run_video(video_url, save_dir)

            else:
                raise FileNotFoundError(f"路径无效: {file_path}")


class YOLO_AUTO(BasePipeline):
    def __init__(self, model_path, conf_thres=0.5, nms_thres=0.5, class_names=None):
        self.inference = Inferencer(model_path)
        input_size, output_size = self.inference.get_info() # rknn无法直接读取
        self.preprocess = PreProcessor_for_YOLO_general(input_size)
        # [[1, 84, 8400]] yolov8n.onnx
        # [[1, 80, 80, 80], [1, 80, 40, 40], [1, 80, 20, 20]] yolov8pose-rknn

        print("\n\033[32m初始化YOLO_AUTO模块, 核心代码为自动选取合适的前后处理\n\033[0m")


class YOLOv5(BasePipeline):
    def init_blocks(self, config=None):
        self.inference = Inferencer(self.model_path)
        self.preprocess = PreProcessor_for_YOLO_general(self.input_size)
        if config is None:
            config = self.model_path.replace('.onnx', '.yaml').replace('.rknn', '.yaml')
        content = open(config, "r", encoding="utf-8").read()
        conf = yaml.safe_load(content)
        masks, anchors = conf["mask"], conf["anchors"]
        self.postprocess = PostProcessor_for_YOLOv5(self.input_size, self.conf_thres, self.nms_thres, masks, anchors)
        self.visualizer = Visualizer(self.class_names)
        print(f"\n\033[32m初始化YOLOv5模块, 核心代码为yolov5 - config文件: {config}\n\033[0m")


class YOLOv8(BasePipeline):
    def init_blocks(self):
        self.inference = Inferencer(self.model_path)
        self.preprocess = PreProcessor_for_YOLO_general(self.input_size)
        self.postprocess = PostProcessor_for_YOLOv8(self.input_size, self.conf_thres, self.nms_thres, self.pose)
        self.visualizer = Visualizer(self.class_names)
        self.tracker = BYTETracker(self.track_thresh,
                            self.track_buffer,
                            self.match_thresh,
                            self.mot20) if self.use_track else None
        print("\n\033[32m初始化YOLOv8模块, 核心代码为yolov8\n\033[0m")


class RopeSkippingPose(BasePipeline):
    def __init__(self,
                 model_path,
                 input_size=(640, 640),
                 conf_thres=0.5,
                 nms_thres=0.5,
                 class_names=["up", "mid-front", "low-front", "low-behind", "below", "other"],
                 max_queue_size=50,
                 model_parallel=12,
                 show=False,
                 ):
        self.model_path = model_path
        self.max_queue_size = max_queue_size
        self.model_parallel = model_parallel
        self.show = show
        
        self.inference = Inferencer(model_path)
        self.preprocess = PreProcessor_for_YOLO_general(input_size)
        self.postprocess = PostProcessor_for_RopeSkipping(input_size, conf_thres, nms_thres)
        self.visualizer = Visualizer(class_names)
        print("\n\033[32m初始化跳绳模块, 核心代码为yolov11-pose 35骨骼点, 且NMS不针对单独类别\n\033[0m")


class YOLOv8_Face(BasePipeline):
    """
    v8 face
    """
    def __init__(self,
                 model_path,
                 input_size=(640, 640),
                 conf_thres=0.5,
                 nms_thres=0.5,
                 class_names=["face"],
                 max_queue_size=50,
                 model_parallel=12,
                 show=False,
                 ):
        self.model_path = model_path
        self.max_queue_size = max_queue_size
        self.model_parallel = model_parallel
        self.show = show

        self.inference = Inferencer(model_path)
        self.preprocess = PreProcessor_for_YOLO_general(input_size)
        self.postprocess = PostProcessor_for_YOLOv8_Face(input_size, conf_thres, nms_thres)
        self.visualizer = Visualizer(class_names)
        print("\n\033[32m初始化YOLOFace模块, 核心代码为yolov8-face 5骨骼点\n\033[0m")


class YOLOv26(BasePipeline):
    """YOLOv26 模型推理流水线组装类"""

    def init_blocks(self) -> None:
        """初始化 YOLOv26 专属组件，包含适配 RKNN 6 输出头的后处理器"""
        self.inference = Inferencer(self.model_path)
        self.preprocess = PreProcessor_for_YOLO_general(self.input_size)

        # 挂载适配 YOLOv26 逻辑的后处理器
        self.postprocess = PostProcessor_for_YOLOv26(
            self.input_size,
            self.conf_thres,
            self.nms_thres
        )

        self.visualizer = Visualizer(self.class_names)

        # 启用跟踪器 (如开启 use_track 参数)
        self.tracker = BYTETracker(self.track_thresh,
                                   self.track_buffer,
                                   self.match_thresh,
                                   self.mot20) if self.use_track else None

        print("\n\033[32m初始化 YOLOv26 模块\n\033[0m")

    # ================= 核心修复代码：移除嵌套的 with 语句 =================

    def run_image(self, image_path, save_path=None):
        """
        重写父类方法：移除内部的 with self 块
        防止在文件夹推理循环中，RKNN 资源在单张图片处理完后被 release
        """
        return self._run_image(image_path, save_path)

    def run_video(self, video_path, save_path):
        """
        重写父类方法：移除内部的 with self 块
        确保处理完视频后，RK3588 的 NPU 运行时不会提前关闭，解决后续文件的 NoneType 报错
        """
        if self.model_parallel > 1:
            return self._run_video_async(video_path, save_path, self.max_queue_size, self.model_parallel, self.show)
        else:
            return self._run_video(video_path, save_path)


