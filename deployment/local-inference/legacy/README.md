# Datature Legacy Inference Scripts

> **Warning**:
> This folder contains legacy scripts for making predictions with Datature's legacy models. These scripts are not actively maintained as our model architectures have been updated. If you have recently exported a model from Datature (after April 2023), please use the scripts in the `deployment/local-inference` folder instead.

The scripts & jupyter notebooks in this folder provide Nexus users a guideline for those who may want to load a model for prediction and integrate them into their codebase or modify some parameters instead of using [Portal](https://github.com/datature/portal) to predict directly.

## Getting Started

Users can download three kinds of model formats from Nexus currently: Tensorflow, TFlite and ONNX.<br>

- [Tensorflow Model - Bounding Box (Direct Download)](#tensorflow-model---bounding-box)
- [Tensorflow Model - Bounding Box (Datature Hub Download)](#tensorflow-model---bounding-box)
- [Tensorflow Model - Masks (Direct Download)](#tensorflow-model---masks)
- [TFLite Model - Bounding Box (Direct Download)](#tflite-model---bounding-box)

### Environment Requirements

python 3.7 =< version =<3.9<br>
Jupyter Notebook<br>
tensorflow >= 2.5.0<br>

<!-- Predict with Different Model -->

## Tensorflow Models

<details open>
     <summary>Click to expand</summary><br>

Under the `scripts/inference/tensorflow/` folder, there are subfolders of different types of computer vision models application or configuration, in `Tensorflow` format:

- [Tensorflow Model - Bounding Box](#tensorflow-model---bounding-box)
- [Tensorflow Model - Masks](#tensorflow-model---masks)

### File Structure

Each of the subfolders contains a standard file structure. Below are descriptions for each file and folder regarding its content or purpose.

- `input/`: Some sample test images for prediction
- `output/`: Output folder to store predicted images
- `requirements.txt`: Python3 dependencies
- `model_architecture/`: All bounding box model architectures offered on Nexus platform
  - `predict.py`: Python script to load downloaded model and obtain predictions
  - `tf_xxx.ipynb`: Jupyter notebook script to load model and obtain predictions
  - `predict_hub.py`: Python script to load model directly from Nexus platform

### Tensorflow Model - Bounding Box

<details>
     <summary>Click to expand</summary><br>

`xxx` in the subsequent command prompts represents chosen model architecture, which is also the name of the folders within the `model_architecture/` folder.

### Command to run Downloaded Model Python script

```bash
cd scripts/inference/tensorflow/bounding_box/
```

```bash
pip install -r requirements.txt
```

```bash
cd model_architecture/xxx
```

```bash
python predict.py --input "path_to_input_folder" --output "path_to_output_folder" --width 640 --height 640 --threshold 0.7 --model "path_to_model" --label "path_to_labelmap"
```

Sample default command

```bash
python predict.py --input "./input" --output "./output" --width 640 --height 640 --threshold 0.7 --model "./saved_model" --label "./label_map.pbtxt"
```

Below is the list of modifiable script parameters and its description.

```bash
--input "path_to_input_folder" (Required)
--output "path_to_output_folder" (Required)
--model "path_to_model" (Required)
--label "path_to_labelmap" (Required)
--width "width of image to load" (Optional) (default: 640)
--height "height of image to load" (Optional) (default: 640)
--threshold "confidence threshold" (Optional) (default: 0.7)
```

### Command to run Jupyter Notebook

```bash
pip install jupyter
```

```bash
python -m notebook tf_xxx.ipynb
```

### Command to run Datature Hub Python script

```bash
cd scripts/inference/tensorflow/bounding_box/
```

```bash
pip install -r requirements.txt
```

```bash
cd model_architecture/xxx
```

```bash
python predict_hub.py --input "path_to_input_folder" --output "path_to_output_folder"  --threshold 0.7 --secret "Project_secret" --key "Your_model_key"
```

Sample default command

```bash
python predict_hub.py --input "./input" --output "./output" --secret "76d97105923491bfa13c84d74eb5457b3b04dceda19ca009d7af111bd7d05344" --key "f2324a0064025c01da8fe3482177a83a"
```

Below is the list of modifiable script parameters and its description.

```bash
--input "Path to folder that contains input images" (Required) (default:"./input")
--output "Path to folder to store predicted images" (Required)(default:"./output")
--threshold "Prediction confidence threshold" (Optional) (default: 0.7)
--secret "Datature Nexus project secret key" (Required)
--key "Datature Nexus model key" (Required)
```

<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>
</details>

### Tensorflow Model - Masks

<details>
     <summary>Click to expand</summary><br>

`xxx` in the subsequent command prompts represents chosen model architecture, which is also the name of the folders within the `model_architecture/` folder.

### Command to run Downloaded Model Python script

```bash
cd scripts/inference/tensorflow/tensorflow/segmentation/
```

```bash
pip install -r requirements.txt
```

```bash
cd model_architecture/xxx
```

```bash
python predict.py --input "path_to_input_folder" --output "path_to_output_folder" --width 1024 --height 1024 --threshold 0.7 --model "path_to_model" --label "path_to_labelmap"
```

Sample default command

```bash
python predict.py --input "./input" --output "./output" --width 1024 --height 1024 --threshold 0.7 --model "./saved_model" --label "./label_map.pbtxt"
```

Below is the list of modifiable script parameters and its description.

```bash
--input "path_to_input_folder" (Required)
--output "path_to_output_folder" (Required)
--model "path_to_model" (Required)
--label "path_to_labelmap" (Required)
--width "width of image to load" (Optional) (default: 1024)
--height "height of image to load" (Optional) (default: 1024)
--threshold "confidence threshold" (Optional) (default: 0.7)
```

### Command to run Jupyter Notebook

```bash
pip install jupyter
```

```bash
python -m notebook tf_xxx.ipynb
```

<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>

</details>
</details>

## TFLite Models

<details open>
     <summary>Click to expand</summary><br>

Under the `scripts/inference/tflite/` folder, there are subfolders of different types of computer vision models application or configuration, in `TFLite` format:

- [TFLite Model - Bounding Box](#tflite-model---bounding-box)

### File Structure

Each of the subfolders contains a standard file structure. Below are descriptions for each file and folder regarding its content or purpose.

- `input/`: Some sample test images for prediction
- `output/`: Output folder to store predicted images
- `requirements.txt`: Python3 dependencies
- `model_architecture/`: All bounding box model architectures offered on Nexus platform
  - `predict.py`: Python script to load downloaded model and obtain predictions
  - `tflite_xxx.ipynb`: Jupyter notebook script to load model and obtain predictions

### TFLite Model - Bounding Box

<details>
     <summary>Click to expand</summary><br>

`xxx` in the subsequent command prompts represents chosen model architecture, which is also the name of the folders within the `model_architecture/` folder.

### Command to run Downloaded Model Python script

```bash
cd scripts/inference/tflite/bounding_box/
```

```bash
pip install -r requirements.txt
```

```bash
cd model_architecture/xxx
```

```bash
python predict.py --input "path_to_input_folder" --output "path_to_output_folder" --width 640 --height 640 --threshold 0.7 --model "path_to_model" --label "path_to_labelmap"
```

Sample default command

```bash
python predict.py --input "./input" --output "./output" --width 640 --height 640 --threshold 0.7 --model "./tf.lite" --label "./label_map.pbtxt"
```

Below is the list of modifiable script parameters and its description.

```bash
--input "path_to_input_folder" (Required)
--output "path_to_output_folder" (Required)
--model "path_to_model" (Required)
--label "path_to_labelmap" (Required)
--width "width of image to load" (Optional) (default: 640)
--height "height of image to load" (Optional) (default: 640)
--threshold "confidence threshold" (Optional) (default: 0.7)
```

### Command to run Jupyter Notebook

```bash
pip install jupyter
```

```bash
python -m notebook tflite_xxx.ipynb
```

<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>

</details>

</details>

## ONNX Models

<details open>
     <summary>Click to expand</summary><br>

Under the `onnx/` folder, there are subfolders of different types of computer vision models application or configuration, in `ONNX` format:

- [ONNX Model - Bounding Box](#tflite-model---bounding-box)

### File Structure

Each of the subfolders contains a standard file structure. Below are descriptions for each file and folder regarding its content or purpose.

- `input/`: Some sample test images for prediction
- `output/`: Output folder to store predicted images
- `requirements.txt`: Python3 dependencies
- `model_architecture/`: All bounding box model architectures offered on Nexus platform
  - `predict.py`: Python script to load downloaded model and obtain predictions

### ONNX Model - Bounding Box

<details>
     <summary>Click to expand</summary><br>

`xxx` in the subsequent command prompts represents chosen model architecture, which is also the name of the folders within the `model_architecture/` folder.

### Command to run Downloaded Model Python script

```bash
cd onnx/bounding_box/
```

```bash
pip install -r requirements.txt
```

```bash
cd model_architecture/xxx
```

```bash
python predict.py --input "path_to_input_folder" --output "path_to_output_folder" --width 640 --height 640 --threshold 0.7 --model "path_to_model" --label "path_to_labelmap"
```

Sample default command

```bash
python predict.py --input "./input" --output "./output" --width 640 --height 640 --threshold 0.7 --model "./tf.lite" --label "./label_map.pbtxt"
```

Below is the list of modifiable script parameters and its description.

```bash
--input "path_to_input_folder" (Required)
--output "path_to_output_folder" (Required)
--model "path_to_model" (Required)
--label "path_to_labelmap" (Required)
--width "width of image to load" (Optional) (default: 640)
--height "height of image to load" (Optional) (default: 640)
--threshold "confidence threshold" (Optional) (default: 0.7)
```

<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>

</details>

</details>

<!-- MARKDOWN LINKS & IMAGES -->
