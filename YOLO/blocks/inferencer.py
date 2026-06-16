import os
import numpy as np
from PIL import Image


class Inferencer:
    # --- 修改点 1：__init__ 增加 layout 参数，并记录 self.layout ---
    def __init__(self, model_path, rknn_core="auto", layout=None):
        """
        Args:
            model_path (str): 模型路径
            rknn_core (str, optional): RKNN使用的核. Defaults to "auto".
            layout (str, optional): 指定布局格式如 "NCHW", 默认 None 不转置
        """
        self.model_path = model_path
        self.rknn_core = rknn_core
        self.layout = layout  # 记录布局格式
        suffix = ""
        if self.model_path.endswith(".rknn"):
            from rknnlite.api import RKNNLite
            if self.rknn_core == "auto":
                self.core_mask = RKNNLite.NPU_CORE_AUTO
                suffix = "RKNN CORE AUTO"
            elif self.rknn_core == "0" or self.rknn_core == 0:
                self.core_mask = RKNNLite.NPU_CORE_0
                suffix = "RKNN CORE 0"
            elif self.rknn_core == "1" or self.rknn_core == 1:
                self.core_mask = RKNNLite.NPU_CORE_1
                suffix = "RKNN CORE 1"
            elif self.rknn_core == "2" or self.rknn_core == 2:
                self.core_mask = RKNNLite.NPU_CORE_2
                suffix = "RKNN CORE 2"
            else:
                raise ValueError(f"不支持的rknn_core: {self.rknn_core}, only support: [0, 1, 2, auto]")
            self.rknn = RKNNLite()

        if self.model_path.endswith(".onnx"):
            self.onnx_session = None

        if self.model_path.endswith(".mlpackage"):
            self.coreml = None

        self.load_model()

        print(f"\n\033[32m初始化YOLO推理类(inferencer.Inferencer) {suffix}\n\033[0m")

    def load_model(self):
        """
        加载模型: 支持ONNX，RKNN，CoreML
        """
        # 判断模型是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found: {self.model_path}, current work dir: {os.getcwd()}, abs path: {os.path.abspath(self.model_path)}")

        # 加载onnx模型
        if self.model_path.endswith(".onnx"):
            import onnxruntime as ort
            self.onnx_session = ort.InferenceSession(self.model_path,
                                                     providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # 加载rknn模型
        elif self.model_path.endswith(".rknn"):
            if self.rknn.load_rknn(self.model_path) != 0:
                raise RuntimeError("RKNN model load failed")

        # 加载coreml模型 
        elif self.model_path.endswith(".mlpackage"):
            import coremltools as ct
            self.coreml = ct.models.MLModel(self.model_path)

        # 加载成功打印
        print(f"\n\033[32m加载模型成功: {os.path.abspath(self.model_path)}\n\033[0m")

    def get_info(self):
        """原有模型分析逻辑，完全保留不做改动"""
        # onnx信息获取
        if self.model_path.endswith(".onnx"):
            inputs_info = self.onnx_session.get_inputs()
            outputs_info = self.onnx_session.get_outputs()
            input_names = [input.name for input in inputs_info]
            output_names = [output.name for output in outputs_info]
            input_shapes = [input.shape for input in inputs_info]
            output_shapes = [output.shape for output in outputs_info]
            print(f"\n\033[32m模型信息分析:\n"
                  f"Model Path: {self.model_path}\n"
                  f"Input Names: {input_names}\n"
                  f"Output Names: {output_names}\n"
                  f"Input Shapes: {input_shapes}\n"
                  f"Output Shapes: {output_shapes}\n\033[0m")
            assert len(input_shapes) == 1, "该onnx模型的输入为多输入, 疑似存在异常"
            assert len(input_shapes[0].shape) == 4, "该onnx模型的输入维度不为[b, c, h, w] 疑似存在异常"
            return input_shapes[0][2:], output_shapes

        # rknn信息获取
        elif self.model_path.endswith(".rknn"):
            return (640, 640), (0, 0)

        # coreml信息获取
        elif self.model_path.endswith("mlpackage"):
            spec = self.coreml.get_spec()
            print("===== Inputs =====")
            for inp in spec.description.input:
                print(f"Name: {inp.name}")
                print(f"Type: {inp.type.WhichOneof('Type')}")

                if inp.type.WhichOneof("Type") == "imageType":
                    print(f"  Image format: {inp.type.imageType.colorSpace}")
                    print(f"  Width x Height: {inp.type.imageType.width} x {inp.type.imageType.height}")
                elif inp.type.WhichOneof("Type") == "multiArrayType":
                    shape = inp.type.multiArrayType.shape
                    print(f"  Shape: {list(shape)}")
            print("===== Outputs =====")
            for out in spec.description.output:
                print(f"Name: {out.name}")
                print(f"Type: {out.type.WhichOneof('Type')}")

                if out.type.WhichOneof("Type") == "multiArrayType":
                    shape = out.type.multiArrayType.shape
                    print(f"  Shape: {list(shape)}")
                print()

    # --- 修改点 2：init_runtime 删除了会导致报错的 self.rknn.query 调用 ---
    def init_runtime(self):
        if self.model_path.endswith(".rknn"):
            if self.rknn.init_runtime(core_mask=self.core_mask) != 0:
                raise RuntimeError("RKNN init failed")
            # 删除了 query 接口以适配板端 RKNNLite

    # --- 修改点 3：norm 针对 RKNN 直接返回 uint8 ---
    def norm(self, input_data):
        if self.model_path.endswith(".rknn"):
            # input_data /= 255
            return input_data

        elif self.model_path.endswith(".onnx"):
            input_data = input_data.astype(np.float32)
            input_data /= 255

        elif self.model_path.endswith(".mlpackage"):
            input_data = input_data[0]

        return input_data

    def infer(self, input_data):
        # infer的输入input_data确保为rgb格式图像

        # rknn 通道转化+归一化+推理
        input_data = self.norm(input_data)

        # onnx推理
        if self.model_path.endswith(".onnx"):
            name = self.onnx_session.get_inputs()[0].name
            output_names = [o.name for o in self.onnx_session.get_outputs()]
            output = self.onnx_session.run(output_names, {name: input_data.transpose(0, 3, 1, 2)})

        # --- 修改点 4：rknn 推理分支改为根据 self.layout 进行智能转置 ---
        elif self.model_path.endswith(".rknn"):
            # 智能适配：仅对 NCHW 格式进行转置
            if self.layout == 2 or self.layout == "NCHW":
                input_data = input_data.transpose(0, 3, 1, 2)

            output = self.rknn.inference(inputs=[input_data])
            if output is None:
                print("\033[31m[错误] NPU 推理失败！\033[0m")
                return []

        # coreml推理
        elif self.model_path.endswith(".mlpackage"):
            name = self.coreml.get_spec().description.input[0].name
            pil_image = Image.fromarray(input_data, mode='RGB')
            output = self.coreml.predict({name: pil_image})
            output = [output[k] for k in output.keys()]

        return output

    def release(self):
        # rknn模型需要单独释放
        if self.model_path.endswith(".rknn"):
            self.rknn.release()