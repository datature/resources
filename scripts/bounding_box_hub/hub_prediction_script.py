"""
Prediction script for trained tensorflow PB model.
"""

import os
import time
import glob
import numpy as np
import tensorflow as tf
import cv2
from dataturehub.hub import HubModel
from dataturehub.utils.visualize import visualize_bbox
from PIL import Image

## Comment out next line to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
## Comment out to set verbose to true
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

PROJECT_SECRET="<YOUR_PROJECT_SECRET>"
MODEL_KEY="<YOUR_MODEL_KEY>"
INPUT_FOLDER = "./Test_Images/"
OUTPUT_FOLDER = "./Outputs"
THRESHOLD = 0.5

def main():
    ## Load argument variables

    hub_model = HubModel(
        project_secret=PROJECT_SECRET,
        model_key=MODEL_KEY,
    )
    trained_model = hub_model.load_tf_model()
    category_index = hub_model.load_label_map()
    ## Run prediction on each image
    for each_image in glob.glob(os.path.join(INPUT_FOLDER, "*")):
        print("Predicting for {}...".format(each_image))
        input_tensor = hub_model.load_image_with_model_dimensions(each_image)
        ## Feed image into model
        detections_output = trained_model(input_tensor)
        visualized_image = visualize_bbox(
            each_image, detections_output, category_index, THRESHOLD
        )

        ## Save predicted image
        filename = os.path.basename(each_image)  # each_image.split("/")[-1]
        image_predict = Image.fromarray(visualized_image)
        image_predict.save(os.path.join(OUTPUT_FOLDER, filename))

        print(
            "Saving predicted images to {}...".format(
                os.path.join(OUTPUT_FOLDER, filename)
            )
        )


if __name__ == "__main__":
    main()
