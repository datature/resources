# YOLOv8 Evaluation Guide by Datature

This guide is for carrying out model evaluations using exported YOLOv8 models trained with Datature. This is useful for comparing model metrics with quantized or pruned models.

## Requirements

To use GPU, the CUDA requirement is >= 11.2.

Some of the package versions specified may require Python < 3.11.

Install the required packages using:

`pip3 install -r requirements.txt`

## Making Predictions

The validation scripts can be run as follows:

```shell
python3 <TASK>_val.py \
    -i val_img_path \
    -m model_path \
    -a val_anno_path \
    -s input_size \
    -l label_path \
    -t threshold \
    --save
```

**TASK** refers to the task type, which can be either `bbox` for object detection, `cls` for classification, `seg` for segmentation, and `pose` for keypoint detection / pose estimation.

**val_img_path** refers to the path to the validation image directory.

**model_path** refers to the path to the YOLOv8 model, the model name is typically prepended with `datature_yolov8*`.

**val_anno_path** refers to the path to the validation annotation file in COCO format.

**input_size** refers to the input size of the model. For example, if the model is trained with input size of 320x320, then the input size should be 320.

**label_path** refers to the path to the label file. This can be found in your exported model zip file, typically named `label.txt` or `label_map.pbtxt`.

**threshold** refers to the threshold value in range (0.0, 1.0) for the prediction score. Only predictions with scores above the threshold value will be shown on the output image.

**\-\-save** is an optional flag to save the output images with predictions drawn on them.

## Documentation

For more information around other training settings, please refer to [Datature's documentation](https://developers.datature.io/docs/model-selection-and-options).
