# Datature-Ultralytics YOLOV8 Model Guide

This guide is for carrying out predictions using exported Datature-Ultralytics YOLOV8 models.

## Requirements

To use GPU, the CUDA requirement is >= 11.2.

Install the required packages using:

`pip3 install -r <MODEL_FORMAT>/requirements.txt`

where `<MODEL_FORMAT>` is the format of the model you are using. For example, if you exported an ONNX model from Nexus, you would run:

`pip3 install -r onnx/requirements.txt`

## Making Predictions

The `predict.py` file can be run as follows:

```shell
python3 predict.py \
    -i input_folder_path \
    -m model_path \
    -t threshold
```

**input_folder_path** refers to the path to the folder where the images for prediction are stored.

**model_path** refers to the path to saved_model (not the saved_model directory)

**threshold** refers to the threshold value in range (0.0, 1.0) for the prediction score. Only predictions with scores above the threshold value will be shown on the output image.
