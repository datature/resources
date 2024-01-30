# YOLOv8 CV2DNN Guide by Datature

This guide is for carrying out predictions using exported YOLOv8 models trained with Datature. The model must be exported in the ONNX format to be used natively with OpenCV's DNN module.

## Requirements

To use GPU, the CUDA requirement is >= 11.2.

Some of the package versions specified may require Python < 3.11.

Install the required packages using:

`pip3 install -r requirements.txt`

## Making Predictions

The `predict.py` file can be run as follows:

```shell
python3 predict.py \
    -i input_folder_path \
    -o output_folder_path \
    -m model_path \
    -s input_size \
    -l label_path \
    -t threshold
```

**input_folder_path** refers to the path to the folder where the images for prediction are stored.

**output_folder_path** refers to the path to the folder where the output images with predictions drawn on them will be saved.

**model_path** refers to the path to the YOLOv8 ONNX model, the model name is typically prepended with `datature_yolov8*` and has the `.onnx` extension.

**input_size** refers to the input size of the model. For example, if the model is trained with input size of 320x320, then the input size should be 320.

**label_path** refers to the path to the label file. This can be found in your exported model zip file, typically named `label.txt` or `label_map.pbtxt`.

**threshold** refers to the threshold value in range (0.0, 1.0) for the prediction score. Only predictions with scores above the threshold value will be shown on the output image.

## Documentation

For more information around other training settings, please refer to [Datature's documentation](https://developers.datature.io/docs/model-selection-and-options).
