<div id="top"></div>

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
<ol>
    <li><a href="#bounding-box-with-tensorflow-model">Bounding Box with Tensorflow Model (Download model derectly)</a></li>
    <li><a href="#bounding-box-hub-with-tensorflow-model">Bounding Box Hub with Tensorflow Model (Access model by datature Hub)</a></li>
    <li><a href="#segmentation-with-tensorflow-model">Segmentation with Tensorflow Model (Download model derectly)</a></li>
    <li><a href="#bounding-box-with-tensorflowlite-model">Bounding Box with TFlite Model (Download model derectly)</a></li>
    <li><a href="#bounding-box-hub-with-tensorflowlite-model">Bounding Box Hub with TFlite Model (Access model by datature Hub)</a></li>
    <li><a href="#segmentation-with-tensorflowlite-model">Segmentation with TFlite Model (Download model derectly)</a></li>
</ol>



### Environment Information
python 3.7<version<3.9<br>
Jupyter Notebook <br>

<!-- Predict with Different Model -->

## Tensorflow Models
<details open>
     <summary>Click to expand</summary>
     
### File Structure
All the three kinds of models have a common file structure.
Description for each file and folder in terms of its content or purpose are shown below.

- input/: Some sample test images for prediction
- output/: Output folder to store predicted images
- saved_model/: Contains trained model ("hub" one donot need)
- labelmap.pbtxt: Label map used for prediction ("hub" one donot need)
- requirements.txt: Contains Python3 dependencies
- prediction_[model].py: Python3 script to run for prediction
- prediction_[model].ipynb: Jupyter notebook script to run for prediction
     
### Bounding Box with Tensorflow Model
<details>
     <summary>Click to expand</summary>
	
#### Command to Run Script in Python3
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

##### Arguments for Python3 File
Parameters below can be modified before prediction.
```
--input "path_to_input_folder" (Required)
--output "path_to_output_folder" (Required)
--size "size of image to load" (Optional) (default: 320x320)
--threshold "confidence threshold" (Optional) (default: 0.7)
--model "path_to_model" (Optional) (default: "./saved_model")
--label "path_to_labelmap" (Optional) (default: "./label_map.pbtxt")
```
#### Command to Run Script in Jupyter Notebook
```
pip install jupyter
```
```
python -m notebook prediction_bb.ipynb
```

<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>
</details>

### Bounding Box Hub with Tensorflow Model
<details>
     <summary>Click to expand</summary>
	
#### Command to Run Script in Python3
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
##### Arguments for Python3 File
```
--input "path_to_input_folder" (Optional) (default:"./input/")
--output "path_to_output_folder" (Optional)(default:"./output")
--threshold "confidence threshold" (Optional) (default: 0.7)
--secret "Project secret" (Required)
--key "Your model key" (Required) 
```
#### Set Up and Running in Jupyter Notebook
First, go to jupyter notebook to change PROJECT_SECRETE and MODUEL_KEY to own one. 
```
pip install jupyter
```
```
python -m notebook prediction_bbhub.ipynb
```
<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>
	
</details>




### Segmentation with Tensorflow Model
<details>
     <summary>Click to expand</summary>
	
#### Command to Run Script in Python3
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

##### Arguments for Python3 File
Parameters below can be modified before prediction.
```
--input "path_to_input_folder" (Required)
--output "path_to_output_folder" (Required)
--size "size of image to load" (Optional) (default: 320x320)
--threshold "confidence threshold" (Optional) (default: 0.7)
--model "path_to_model" (Optional) (default: "./saved_model")
--label "path_to_labelmap" (Optional) (default: "./label_map.pbtxt")
```
#### Command to Run Script in Jupyter Notebook
```
pip install jupyter
```
```
python -m notebook prediction_seg.ipynb
```	
	
<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>
	
</details>
</details>




## TensorflowLITE Models
<details open>
     <summary>Click to expand</summary>
     
### File Structure
All the three kinds of models have a common file structure.
Description for each file and folder in terms of its content or purpose are shown below.

- input/: Some sample test images for prediction
- output/: Output folder to store predicted images
- tf.lite: Trained model ("hub" one donot need)
- labelmap.pbtxt: Label map used for prediction ("hub" one donot need)
- requirements.txt: Contains Python3 dependencies
- prediction_[model]lite.py: Python3 script to run for prediction
- prediction_[model]lite.ipynb: Jupyter notebook script to run for prediction
     
### Bounding Box with Tensorflowlite Model
<details>
     <summary>Click to expand</summary>
	
#### Command to Run Script in Python3
```
cd tensorflowlite_scripts/bounding_box
```

```
pip install -r requirements.txt
```

```
python prediction_bblite.py --INPUT "path_to_input_folder" --OUTPUT "path_to_output_folder" --SIZE [640,640] --THRESHOLD 0.7 --MODEL "path_to_model" --LABEL "path_to_labelmap"
```

Example Default Command
```
python prediction_bblite.py --INPUT "./input" --OUTPUT "./output" --SIZE [640,640] --THRESHOLD 0.7 --MODEL "./tf.lite" --LABEL "./label_map.pbtxt"
```

##### Arguments for Python3 File
Parameters below can be modified before prediction.
```
--INPUT "path_to_input_folder" (Optional) (default:"./input")
--OUTPUT "path_to_output_folder" (Optional) (default:"./output")
--SIZE "size of image to load" (Optional) (default: [640,640])
--THRESHOLD "confidence threshold" (Optional) (default: 0.7)
--MODEL "path_to_model" (Optional) (default: "./tf.lite")
--LABEL "path_to_labelmap" (Optional) (default: "./label_map.pbtxt")
```
#### Command to Run Script in Jupyter Notebook
```
pip install jupyter
```
```
python -m notebook prediction_bblite.ipynb
```
	
<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>
	
</details>

### Bounding Box Hub with Tensorflowlite Model
<details>
     <summary>Click to expand</summary>
	
<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>
	
</details>




### Segmentation with Tensorflowlite Model
<details>
     <summary>Click to expand</summary>
	
<br/>
<div align="right">
    <b><a href="#top">↥ back to top</a></b>
</div>
<br/>
	
</details>
</details>





<!-- MARKDOWN LINKS & IMAGES -->

