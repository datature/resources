# Local Inference

The scripts & jupyter notebooks in this folder provide Nexus users a guideline for those who may want to load a model for prediction and integrate them into their codebase or modify some parameters instead of using [Portal](https://github.com/datature/portal) to predict directly.

## Getting Started

Each folder for the various model formats is structured accordingly:

- `input/`: Some sample test images for prediction
- `output/`: Output folder to store predicted images
- `requirements.txt`: Python3 dependencies
- `<bound_type>_<model_format>.py`: Python script to load model, run inference, and overlay predictions on images
- `README.md`: Instructions for running the script

### Environment Requirements

> **Note**
> We recommend using a virtual environment to install the dependencies. For more information, see [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html).

- python 3.7 =< version

## Model Formats

Users can currently run inference on the following model formats from Nexus in the tables below:

### Object Detection Models

| Model Format | Model Architecture(s) |
| :--- | :--- |
| [Tensorflow](./tensorflow/bounding_box/README.md) | Faster R-CNN, ResNet, EfficientDet, MobileNet, YOLOv4 |
| [TFLite](./tflite/bounding_box/README.md) | Faster R-CNN, ResNet, EfficientDet, MobileNet, YOLOv4 |
| [ONNX](./onnx/bounding_box/README.md) | Faster R-CNN, ResNet, EfficientDet, MobileNet, YOLOv4 |
| [PyTorch](./pytorch/bounding_box/README.md) | Faster R-CNN, ResNet, EfficientDet, MobileNet, YOLOv4 |

### Instance Segmentation Models

| Model Format | Model Architecture(s) |
| :--- | :--- |
| [Tensorflow](./tensorflow/segmentation/instance/README.md) | Mask R-CNN |
| [TFLite](./tflite/segmentation/instance/README.md) | Mask R-CNN |
| [ONNX](./onnx/segmentation/instance/README.md) | Mask R-CNN |

### Semantic Segmentation Models

| Model Format | Model Architecture(s) |
| :--- | :--- |
| [Tensorflow](./tensorflow/segmentation/semantic/README.md) | DeepLabV3, U-Net, FCN |
| [TFLite](./tflite/segmentation/semantic/README.md) | DeepLabV3, U-Net, FCN |
| [ONNX](./onnx/segmentation/semantic/README.md) | DeepLabV3, U-Net, FCN |
| [PyTorch](./pytorch/segmentation/semantic/README.md) | DeepLabV3, U-Net, FCN |
