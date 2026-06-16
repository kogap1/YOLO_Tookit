from setuptools import setup, find_packages

install_requires = [
    "numpy>=1.23.0",
    "pillow>=11.1.0",
    "opencv-python>=4.11.0.0",
    "scipy>=1.17.0",
    "lap>=0.5.12",
    "cython_bbox>=0.1.5",
    "onnxruntime>=1.19.2"
]

setup(
    name="YOLO",
    version="1.0.3",
    description="YOLO toolkit",
    author="YOLO Toolkit Contributors",
    packages=find_packages(),     # ⭐ 关键：找到 YOLO/ 包
    install_requires=install_requires,
    python_requires=">=3.8",
)
