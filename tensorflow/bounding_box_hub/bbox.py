"""Prediction script for trained tensorflow PB model.
"""

import os
import time
import glob
import argparse
import numpy as np
import tensorflow as tf
import cv2
from datature_hub.hub import HubModel
from datature_hub.utils.visualize import visualize_bbox
from PIL import Image

## Comment out next line to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
## Comment out to set verbose to true
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def args_parser():
    parser = argparse.ArgumentParser(
        description="Datature Open Source Prediction Script"
    )
    parser.add_argument(
        "--input", help="Path to folder that contains input images", default = "./input/"
    )
    parser.add_argument(
        "--output", help="Path to folder to store predicted images", default="./output"
    )
    parser.add_argument(
        "--threshold", help="Prediction confidence threshold", default=0.7
    )
    parser.add_argument(
        "--secret", help="Project secret",  required=True
    )
    parser.add_argument(
        "--key", help="Your model key", required=True
    )
    return parser.parse_args()


# Load argument variables
args = args_parser()
project_secret=args.secret
model_key=args.key
input_folder =args.input
output_folder =args.output
threshold =args.threshold


hub_model = HubModel(
    project_secret=project_secret,
    model_key=model_key,
  )
trained_model = hub_model.load_tf_model()
category_index = hub_model.load_label_map()
# Run prediction on each image
all_detections = []
all_images = glob.glob(os.path.join(input_folder, "*"))
for each_image in all_images:
  print("Predicting for {}...".format(each_image))
  input_tensor = hub_model.load_image_with_model_dimensions(each_image)
  detections_output = trained_model(input_tensor)
  all_detections.append(detections_output)
  
# Save predicted image
for each_image, each_detection in zip(all_images, all_detections):
  visualized_image = visualize_bbox(each_image, each_detection, category_index, threshold)
  filename = os.path.basename(each_image)
  image_predict = Image.fromarray(visualized_image)
  image_predict.save(os.path.join(output_folder, filename))

  print(
    "Saving predicted images to {}...".format(
      os.path.join(output_folder, filename)
    )
  )

