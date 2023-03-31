# Datature ONNX Instance Segmentation Model Guide

This guide is for carrying out predictions using exported Datature ONNX Instance Segmentation models.


## Requirements

To use GPU, the CUDA requirement is >= 11.4. 

Install the required packages using:

`pip3 install -r requirements.txt`


## Making Predictions

The `predict.py` file can be run as follows:

```shell
python3 predict.py \
    -i input_folder_path \
    -o output_folder_path \
    -m model_path \
    -l label_map_path \
    -t threshold 
```

**input_folder_path** refers to the path to the folder where the images for prediction are stored.

**output_folder_path** refers to the path to the folder where the output images after prediction are to be stored. Do note that the output image names will be the same as the input image names so this should not be the same folder as the input folder.

**model_path** refers to the path to saved_model (not the saved_model directory)

**label_map_path** refers to the path to the label map file (not the label map file directory)

**threshold** refers to the threshold value in range (0.0, 1.0) for the prediction score. Only predictions with scores above the threshold value will be shown on the output image. 