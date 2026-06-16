# 模块说明
* `preprocessor.py` 包含NMS操作。在最新版本中将**Norm**、**通道转化**操作移动到到`inferencer.py`的初始化中，以应对同一模型的不同平台部署可能用到的不同前处理。
* `inferencer.py` 包含**加载模型**、**Norm**、**通道转化**、**推理**等操作。