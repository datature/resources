<div id="top"></div>

# Datature Code and Resources Hub

A repository of resources used in our tutorials and guides ⚡️

<!-- INTRODUCTION -->

The scripts & jupyter notebooks in this repository provide Nexus users a guideline for those who may want to load a model for prediction and integrate them into their codebase or modify some parameters instead of using Portal to predict directly.

<!-- GETTING STARTED -->

## Getting Started

Firstly, users should clone this repository and cd to the resource folder.<br>

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

Users can download their trained model directly from Nexus or port the trained model through Datature Hub. Users need two sets of keys for the second method: `Model Key` and `Project Secret Key`.<br>

#### Model Key

To convert that artifact into an exported model for the prediction service, in Nexus, select `Artifacts` under Project Overview. Within the artifacts page, select your chosen artifact and model format to generate a model key for deployment by clicking the triple dots box shown below.

![modelkey](/assets/modelkey.png)

#### Project Secret Key

You can generate the project secret key on the Nexus platform by going to `API Management` and selecting the `Generate New Secret` button, as shown below.

![projectsecret](/assets/projectsecret.png)

### Environment Requirements

python 3.7 =< version =<3.9<br>
Jupyter Notebook<br>
tensorflow == 2.5.0<br>

<!-- Predict with Different Model -->

## Tensorflow Models

<details open>
     <summary>Click to expand</summary>
     
### File Structure
All three different kinds of Tensorflow options have a standard file structure. Below are descriptions for each file and folder regarding its content or purpose.

- `input/`: Some sample test images for prediction
- `output/`: Output folder to store predicted images
- `saved_model/`: Contains sample trained model (`hub` does not need one)
- `label_map.pbtxt`: Contains sample label map (`hub` does not need one)
- `requirements.txt`: Python3 dependencies
- `tf_xxx.py`: Script to load model and obtain predictions
- `tf_xxx.ipynb`: Jupyter notebook script to load model and obtain predictions

### Tensorflow Model - Bounding Box (Direct Download)

<details>
     <summary>Click to expand</summary>

<br>Command to run script

```
cd tensorflow/bounding_box
```

```
pip install -r requirements.txt
```

```
python tf_bbox.py --input "path_to_input_folder" --output "path_to_output_folder" --size "640x640" --threshold 0.7 --model "path_to_model" --label "path_to_labelmap"
```

Sample default command

```
python tf_bbox.py --input "./input" --output "./output" --size "640x640" --threshold 0.7 --model "./saved_model" --label "./label_map.pbtxt"
```

Below is the list of modifiable script parameters and its description.

```
--input "path_to_input_folder" (Required) (default:"./input")
--output "path_to_output_folder" (Required) (default:"./output")
--size "size of image to load" (Optional) (default: 320x320)
--threshold "confidence threshold" (Optional) (default: 0.7)
--model "path_to_model" (Optional) (default: "./saved_model")
--label "path_to_labelmap" (Optional) (default: "./label_map.pbtxt")
```

Command to run Jupyter Notebook

```
pip install jupyter
```

```
python -m notebook tf_bbox.ipynb
```

<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>
</details>

### Tensorflow Model - Bounding Box (Datature Hub Download)

<details>
     <summary>Click to expand</summary>
	
<br>Command to Run Script in Python3
```
cd tensorflow/bounding_box_hub
```

```
pip install -r requirements.txt
```

```
python tf_bbox.py --input "path_to_input_folder" --output "path_to_output_folder"  --threshold 0.7 --secret "Project_secret" --key "Your_model_key"
```

Sample default command

```
python tf_bbox.py --input "./input" --output "./output" --secret "76d97105923491bfa13c84d74eb5457b3b04dceda19ca009d7af111bd7d05344" --key "f2324a0064025c01da8fe3482177a83a"
```

Below is the list of modifiable script parameters and its description.

```
--input "Path to folder that contains input images" (Required) (default:"./input")
--output "Path to folder to store predicted images" (Required)(default:"./output")
--threshold "Prediction confidence threshold" (Optional) (default: 0.7)
--secret "Datature Nexus project secret key" (Required)
--key "Datature Nexus model key" (Required)
```

Change `PROJECT_SECRET` & `MODEL_KEY` settings inside the Jupyter Notebook before running it with the command below.

```
pip install jupyter
```

```
python -m notebook tf_bbox.ipynb
```

<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>
	
</details>

### Tensorflow Model - Masks (Direct Download)

<details>
     <summary>Click to expand</summary>
	
<br>Command to Run Script in Python3
```
cd tensorflow/segmentation
```

```
pip install -r requirements.txt
```

```
python tf_seg.py --input "path_to_input_folder" --output "path_to_output_folder" --size "640x640" --threshold 0.7 --model "path_to_model" --label "path_to_labelmap"
```

Sample default command

```
python tf_seg.py --input "./input" --output "./output" --size "640x640" --threshold 0.7 --model "./saved_model" --label "./label_map.pbtxt"
```

Below is the list of modifiable script parameters and its description.

```
--input "path_to_input_folder" (Required)
--output "path_to_output_folder" (Required)
--size "size of image to load" (Optional) (default: 320x320)
--threshold "confidence threshold" (Optional) (default: 0.7)
--model "path_to_model" (Optional) (default: "./saved_model")
--label "path_to_labelmap" (Optional) (default: "./label_map.pbtxt")
```

Command to run Jupyter Notebook

```
pip install jupyter
```

```
python -m notebook tf_seg.ipynb
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
     <summary>Click to expand</summary>
     
### File Structure
Below are descriptions for each file and folder regarding its content or purpose.

- `input/`: Some sample test images for prediction
- `output/`: Output folder to store predicted images
- `tf.lite`: Contains sample trained model (`hub` does not need one)
- `label_map.pbtxt`: Contains sample label map (`hub` does not need one)
- `requirements.txt`: Python3 dependencies
- `tflite_xxx.py`: Script to load model and obtain predictions
- `tflite_xxx.ipynb`: Jupyter notebook script to load model and obtain predictions

### TFLite Model - Bounding Box (Direct Download)

<details>
     <summary>Click to expand</summary>
	
#### Command to Run Script in Python3
```
cd tflite/bounding_box
```

```
pip install -r requirements.txt
```

```
python tflite_bbox.py --input "path_to_input_folder" --output "path_to_output_folder" --size "640x640" --threshold 0.7 --model "path_to_model" --label "path_to_labelmap"
```

Sample default command

```
python tflite_bbox.py --input "./input" --output "./output" --size "640x640" --threshold 0.7 --model "./tf.lite" --label "./label_map.pbtxt"
```

Below is the list of modifiable script parameters and its description.

```
--input "path_to_input_folder" (Required) (default:"./input")
--OUTPUT "path_to_output_folder" (Required) (default:"./output")
--SIZE "size of image to load" (Optional) (default: 320x320)
--THRESHOLD "confidence threshold" (Optional) (default: 0.7)
--MODEL "path_to_model" (Optional) (default: "./tf.lite")
--LABEL "path_to_labelmap" (Optional) (default: "./label_map.pbtxt")
```

Command to run Jupyter Notebook

```
pip install jupyter
```

```
python -m notebook tflite_bbox.ipynb
```

<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>
	
</details>

</details>

<!-- MARKDOWN LINKS & IMAGES -->
