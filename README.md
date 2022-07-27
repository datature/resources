[![Join Datature Slack](https://img.shields.io/badge/Join%20The%20Community-Datature%20Slack-blueviolet)](https://datature.io/community)

<div id="top"></div>

# Datature Code and Resources Hub

A repository of resources used in our tutorials and guides ⚡️

<!-- INTRODUCTION -->

The scripts & jupyter notebooks in this repository provide Nexus users a guideline for those who may want to load a model for prediction and integrate them into their codebase or modify some parameters instead of using [Portal](https://github.com/datature/portal) to predict directly.

<!-- GETTING STARTED -->

## Getting Started

Firstly, users should clone this repository and change to the resource folder directory.<br>

```
git clone https://github.com/pxdn323/resources.git
```

<br>
Users can download two kinds of model formats from Nexus currently: Tensorflow model and TFlite model.<br><br>

- [Tensorflow Model - Bounding Box (Direct Download)](#tensorflow-model---bounding-box-direct-download)
- [Tensorflow Model - Bounding Box (Datature Hub Download)](#tensorflow-model---bounding-box-datature-hub-download)
- [Tensorflow Model - Masks (Direct Download)](#tensorflow-model---masks-direct-download)
- [TFLite Model - Bounding Box (Direct Download)](#tflite-model---bounding-box-direct-download)

### Download Model

Users can download their trained model directly from [Nexus](https://nexus.datature.io/) or port the trained model through Datature Hub. Users need two sets of keys for the second method: `Model Key` and `Project Secret Key`.<br>

#### Model Key

To convert that artifact into an exported model for the prediction service, in Nexus, select `Artifacts` under Project Overview. Within the artifacts page, select your chosen artifact and model format to generate a model key for deployment by clicking the triple dots box shown below.

![modelkey](/assets/modelkey.png)

#### Project Secret Key

You can generate the project secret key on the Nexus platform by going to `API Management` and selecting the `Generate New Secret` button, as shown below.

![projectsecret](/assets/projectsecret.png)

### Environment Requirements

python 3.7 =< version =<3.9<br>
Jupyter Notebook<br>
tensorflow >= 2.5.0<br>

<!-- Predict with Different Model -->

## Tensorflow Models

<details open>
     <summary>Click to expand</summary><br>

Under the `tensorflow/` folder, there are subfolders of different types of computer vision models application or configuration, in `Tensorflow` format:

- [Tensorflow Model - Bounding Box](#tensorflow-model---bounding-box)
- [Tensorflow Model - Masks](#tensorflow-model---masks)

### File Structure

Each of the subfolders contains a standard file structure. Below are descriptions for each file and folder regarding its content or purpose.

- `input/`: Some sample test images for prediction
- `output/`: Output folder to store predicted images
- `requirements.txt`: Python3 dependencies
- `model_architecture/`: All bounding box model architectures offered on Nexus platform
  - `tf_xxx.py`: Python script to load downloaded model and obtain predictions
  - `tf_xxx.ipynb`: Jupyter notebook script to load model and obtain predictions
  - `tf_hub_xxx.py`: Python script to load model directly from Nexus platform

### Tensorflow Model - Bounding Box

<details>
     <summary>Click to expand</summary><br>

`xxx` in the subsequent command prompts represents chosen model architecture, which is also the name of the folders within the `model_architecture/` folder.

### Command to run Downloaded Model Python script

```
cd tensorflow/bounding_box/
```

```
pip install -r requirements.txt
```

```
cd model_architecture/xxx
```

```
python tf_xxx.py --input "path_to_input_folder" --output "path_to_output_folder" --width 640 --height 640 --threshold 0.7 --model "path_to_model" --label "path_to_labelmap"
```

Sample default command

```
python tf_xxx.py --input "./input" --output "./output" --width 640 --height 640 --threshold 0.7 --model "./saved_model" --label "./label_map.pbtxt"
```

Below is the list of modifiable script parameters and its description.

```
--input "path_to_input_folder" (Required)
--output "path_to_output_folder" (Required)
--model "path_to_model" (Required)
--label "path_to_labelmap" (Required)
--width "width of image to load" (Optional) (default: 640)
--height "height of image to load" (Optional) (default: 640)
--threshold "confidence threshold" (Optional) (default: 0.7)
```

### Command to run Jupyter Notebook

```
pip install jupyter
```

```
python -m notebook tf_xxx.ipynb
```

### Command to run Datature Hub Python script

```
cd tensorflow/bounding_box/
```

```
pip install -r requirements.txt
```

```
cd model_architecture/xxx
```

```
python tf_hub_xxx.py --input "path_to_input_folder" --output "path_to_output_folder"  --threshold 0.7 --secret "Project_secret" --key "Your_model_key"
```

Sample default command

```
python tf_hub_xxx.py --input "./input" --output "./output" --secret "76d97105923491bfa13c84d74eb5457b3b04dceda19ca009d7af111bd7d05344" --key "f2324a0064025c01da8fe3482177a83a"
```

Below is the list of modifiable script parameters and its description.

```
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

```
cd tensorflow/segmentation/
```

```
pip install -r requirements.txt
```

```
cd model_architecture/xxx
```

```
python tf_xxx.py --input "path_to_input_folder" --output "path_to_output_folder" --width 1024 --height 1024 --threshold 0.7 --model "path_to_model" --label "path_to_labelmap"
```

Sample default command

```
python tf_xxx.py --input "./input" --output "./output" --width 1024 --height 1024 --threshold 0.7 --model "./saved_model" --label "./label_map.pbtxt"
```

Below is the list of modifiable script parameters and its description.

```
--input "path_to_input_folder" (Required)
--output "path_to_output_folder" (Required)
--model "path_to_model" (Required)
--label "path_to_labelmap" (Required)
--width "width of image to load" (Optional) (default: 1024)
--height "height of image to load" (Optional) (default: 1024)
--threshold "confidence threshold" (Optional) (default: 0.7)

```

### Command to run Jupyter Notebook

```
pip install jupyter
```

```
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

Under the `tflite/` folder, there are subfolders of different types of computer vision models application or configuration, in `TFLite` format:

- [TFLite Model - Bounding Box](#tflite-model---bounding-box)

### File Structure

Each of the subfolders contains a standard file structure. Below are descriptions for each file and folder regarding its content or purpose.

- `input/`: Some sample test images for prediction
- `output/`: Output folder to store predicted images
- `requirements.txt`: Python3 dependencies
- `model_architecture/`: All bounding box model architectures offered on Nexus platform
  - `tflite_xxx.py`: Python script to load downloaded model and obtain predictions
  - `tflite_xxx.ipynb`: Jupyter notebook script to load model and obtain predictions

### TFLite Model - Bounding Box

<details>
     <summary>Click to expand</summary><br>

`xxx` in the subsequent command prompts represents chosen model architecture, which is also the name of the folders within the `model_architecture/` folder.

### Command to run Downloaded Model Python script

```
cd tflite/bounding_box/
```

```
pip install -r requirements.txt
```

```
cd model_architecture/xxx
```

```
python tflite_xxx.py --input "path_to_input_folder" --output "path_to_output_folder" --width 640 --height 640 --threshold 0.7 --model "path_to_model" --label "path_to_labelmap"
```

Sample default command

```
python tflite_xxx.py --input "./input" --output "./output" --width 640 --height 640 --threshold 0.7 --model "./tf.lite" --label "./label_map.pbtxt"
```

Below is the list of modifiable script parameters and its description.

```
--input "path_to_input_folder" (Required)
--output "path_to_output_folder" (Required)
--model "path_to_model" (Required)
--label "path_to_labelmap" (Required)
--width "width of image to load" (Optional) (default: 640)
--height "height of image to load" (Optional) (default: 640)
--threshold "confidence threshold" (Optional) (default: 0.7)
```

### Command to run Jupyter Notebook

```
pip install jupyter
```

```
python -m notebook tflite_xxx.ipynb
```

<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>
	
</details>

</details>

<!-- MARKDOWN LINKS & IMAGES -->
