# resources
A repository of resources used in our tutorials and guides ⚡️



<!-- INTRODUCTION -->
This repository is provided to Nexus users who may want to load model in to their own code or modify some parameters instead of using Portal to predict directly. 


<!-- GETTING STARTED -->
## Getting Started

### File Structure

Description for each file and folder in terms of its content or purpose.

- input/: Some sample test images for prediction
- output/: Output folder to store predicted images
- saved_model/: Contains trained Tensorflow Model
- labelmap.pbtxt: Label map used for prediction
- requirements.txt: Contains Python3 dependencies
- prediction.py: Script to run for prediction

## Bounding Box
### Command to Run Script in Python3

```
python3 prediction.py --input "path_to_input_folder" --output "path_to_output_folder" --size "640x640" --threshold 0.7 --model "path_to_model" --label "path_to_labelmap"
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





## Bounding Box Hub
### Command to Run Script in Python3

### Arguments for Python3 File

### Set Up and Running in Jupyter Notebook





## Segmentation
### Command to Run Script in Python3

### Arguments for Python3 File

### Set Up and Running in Jupyter Notebook












<!-- MARKDOWN LINKS & IMAGES -->

