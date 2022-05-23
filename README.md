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

Description for each file and folder in terms of its content or purpose.

- input/: Some sample test images for prediction
- output/: Output folder to store predicted images
- saved_model/: Contains trained model
- labelmap.pbtxt: Label map used for prediction
- requirements.txt: Contains Python3 dependencies
- prediction.py: Script to run for prediction

## Bounding Box with Tensorflow Model
### Command to Run Script in Python3
```
pip install -r requirements.txt
```

```
python3 prediction.py --input "path_to_input_folder" --output "path_to_output_folder" --size "640x640" --threshold 0.7 --model "path_to_model" --label "path_to_labelmap"
```
Example Default Command
```
python3 prediction.py --input "./input" --output "./output" --size "640x640" --threshold 0.7 --model "./saved_model" --label "./label_map.pbtxt"
```
### Arguments for Python3 File

```
--input "path_to_input_folder" (Required)
--output "path_to_output_folder" (Required)
--size "size of image to load" (Optional) (default: 320x320)
--threshold "confidence threshold" (Optional) (default: 0.7)
--model "path_to_model" (Optional) (default: "./saved_model")
--label "path_to_labelmap" (Optional) (default: "./label_map.pbtxt")
```
### Set Up and Running in Jupyter Notebook





## Bounding Box Hub with Tensorflow Model
### Command to Run Script in Python3

### Arguments for Python3 File

### Set Up and Running in Jupyter Notebook





## Segmentation with Tensorflow Model
### Command to Run Script in Python3

### Arguments for Python3 File

### Set Up and Running in Jupyter Notebook












<!-- MARKDOWN LINKS & IMAGES -->

