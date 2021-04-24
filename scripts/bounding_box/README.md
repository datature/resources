## Datature Open Source Prediction Script

This folder contains script to run prediction for Datature's trained model

### File Structure

Description for each file and folder in terms of its content or purpose.

- input/: Some sample test images for prediction
- output/: Output folder to store predicted images
- saved_model/: Contains trained Tensorflow Model
- labelmap.pbtxt: Label map used for prediction
- requirements.txt: Contains Python3 dependencies
- prediction.py: Script to run for prediction

### Command to Run Script

```
python3 prediction.py --input "path_to_input_folder" --output "path_to_output_folder" --size "640x640" --threshold 0.7 --model "path_to_model" --label "path_to_labelmap"
```

### Arguments

```
--input "path_to_input_folder" (Required)
--output "path_to_output_folder" (Required)
--size "size of image to load" (Optional) (default: 320x320)
--threshold "confidence threshold" (Optional) (default: 0.7)
--model "path_to_model" (Optional) (default: "./saved_model")
--label "path_to_labelmap" (Optional) (default: "./label_map.pbtxt")
```
