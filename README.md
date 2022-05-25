# resources
A repository of resources used in our tutorials and guides ⚡️



<!-- INTRODUCTION -->
This repository is provided to Nexus users who may want to load model in to their own code or modify some parameters instead of using Portal to predict directly. 


<!-- GETTING STARTED -->
## Getting Started
There are two kinds of model can be downloaded from Nexus: TensorFlow model and TFlite model. Users can download model derectly from Nexus or port the trained model
through Datature Hub. For the second method, users should get two sets of Keys: Model Key and Project Secret Key in advance.
Firstly, users should clone this repository and cd to resource folder.

The usage of the following six different models though both python3 and jupyter notebook will be introduced：

* Bounding Box with Tensorflow Model (Download model derectly)
* Bounding Box Hub with Tensorflow Model (Access model by datature Hub)
* Segmentation with Tensorflow Model (Download model derectly)
* Bounding Box with TFlite Model (Download model derectly)
* Bounding Box Hub with TFlite Model (Access model by datature Hub)
* Segmentation with TFlite Model (Download model derectly)

### File Structure
All the six kinds of models have a common file structure.
Description for each file and folder in terms of its content or purpose are shown below.

- input/: Some sample test images for prediction
- output/: Output folder to store predicted images
- saved_model/: Contains trained model ("hub" one donot need)
- labelmap.pbtxt: Label map used for prediction ("hub" one donot need)
- requirements.txt: Contains Python3 dependencies
- prediction_[model].py: Python3 script to run for prediction
- prediction_[model].ipynb: Jupyter notebook script to run for prediction

### Environment Information
python 3.7<version<3.9<br>
Anaconda Navagator <br>
Jupyter Notebook <br>
If python version in Jupyter Notebook not in range(3.7,3.9),change the version in Environments in ANACONDA,NAVIGATOR
<!-- Predict with Different Model -->
## Bounding Box with Tensorflow Model
### Command to Run Script in Python3
```
cd tensorflow_scripts/bounding_box
```

```
pip install -r requirements.txt
```

```
python prediction_bb.py --input "path_to_input_folder" --output "path_to_output_folder" --size "640x640" --threshold 0.7 --model "path_to_model" --label "path_to_labelmap"
```

Example Default Command
```
python prediction_bb.py --input "./input" --output "./output" --size "640x640" --threshold 0.7 --model "./saved_model" --label "./label_map.pbtxt"
```

#### Arguments for Python3 File
Parameters below can be modified before prediction.
```
--input "path_to_input_folder" (Required)
--output "path_to_output_folder" (Required)
--size "size of image to load" (Optional) (default: 320x320)
--threshold "confidence threshold" (Optional) (default: 0.7)
--model "path_to_model" (Optional) (default: "./saved_model")
--label "path_to_labelmap" (Optional) (default: "./label_map.pbtxt")
```
### Command to Run Script in Jupyter Notebook
```
pip install jupyter
```
```
python -m notebook prediction_bb.ipynb
```



## Bounding Box Hub with Tensorflow Model
### Command to Run Script in Python3
```
cd tensorflow_scripts/bounding_box_hub
```

```
pip install -r requirements.txt
```

```
python prediction_bbhub.py --input "path_to_input_folder" --output "path_to_output_folder"  --threshold 0.7 --secret "Project_secret" --key "Your_model_key"
```

Example Default Command
```
python prediction_bbhub.py  --secret "76d97105923491bfa13c84d74eb5457b3b04dceda19ca009d7af111bd7d05344" --key "f2324a0064025c01da8fe3482177a83a"
```
#### Arguments for Python3 File
```
--input "path_to_input_folder" (Optional) (default:"./input/")
--output "path_to_output_folder" (Optional)(default:"./output")
--threshold "confidence threshold" (Optional) (default: 0.7)
--secret "Project secret" (Required)
--key "Your model key" (Required) 
```
### Set Up and Running in Jupyter Notebook
First, go to jupyter notebook to change PROJECT_SECRETE and MODUEL_KEY to own one. 
```
pip install jupyter
```
```
python -m notebook prediction_bbhub.ipynb
```





## Segmentation with Tensorflow Model
### Command to Run Script in Python3
```
cd tensorflow_scripts/segmentation
```

```
pip install -r requirements.txt
```

```
python prediction_seg.py --input "path_to_input_folder" --output "path_to_output_folder" --size "640x640" --threshold 0.7 --model "path_to_model" --label "path_to_labelmap"
```

Example Default Command
```
python prediction_seg.py --input "./input" --output "./output" --size "640x640" --threshold 0.7 --model "./saved_model" --label "./label_map.pbtxt"
```

#### Arguments for Python3 File
Parameters below can be modified before prediction.
```
--input "path_to_input_folder" (Required)
--output "path_to_output_folder" (Required)
--size "size of image to load" (Optional) (default: 320x320)
--threshold "confidence threshold" (Optional) (default: 0.7)
--model "path_to_model" (Optional) (default: "./saved_model")
--label "path_to_labelmap" (Optional) (default: "./label_map.pbtxt")
```
### Command to Run Script in Jupyter Notebook
```
pip install jupyter
```
```
python -m notebook prediction_seg.ipynb
```

     









<!-- MARKDOWN LINKS & IMAGES -->

