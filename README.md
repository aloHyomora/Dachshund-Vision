# Dachshund-Vision: Visual Sensing Module
Visual Sensing Module for [Dachshund Engine](https://github.com/aloHyomora/Dachshund-Engine) (Human/Object Detection using TensorRT)

## 설치 (간단)
Conda로 Python 3.8 환경을 만들고 OpenCV, pyzmq, ultralytics를 설치합니다:

```bash
conda create -n dachshund-vision python=3.8 -y
conda activate dachshund-vision

# OpenCV, pyzmq, ultralytics (pip)
pip install opencv-python pyzmq ultralytics
```

필요 시 GPU/CUDA용 PyTorch 및 TensorRT는 환경에 맞춰 별도 설치하세요.
