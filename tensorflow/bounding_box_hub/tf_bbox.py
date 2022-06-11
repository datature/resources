"""Prediction script for trained tensorflow PB model using Datature Hub.
"""

import os
import time
import glob
import argparse
import absl.logging

from PIL import Image
from datature_hub.hub import HubModel
from datature_hub.utils.visualize import visualize_bbox

## Disable unnecessary warnings for tensorflow
absl.logging.set_verbosity(absl.logging.ERROR)
## Comment out next line to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
## Comment out to set verbose to true
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def args_parser():
    parser = argparse.ArgumentParser(
        description="Datature Open Source Prediction Script"
    )
    parser.add_argument(
        "--input",
        help="Path to folder that contains input images",
        default="./input/",
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Path to folder to store predicted images",
        default="./output",
        required=True,
    )
    parser.add_argument(
        "--threshold", help="Prediction confidence threshold", default=0.7
    )
    parser.add_argument(
        "--secret", help="Datature Nexus project secret key", required=True
    )
    parser.add_argument(
        "--key", help="Datature Nexus model key", required=True
    )
    return parser.parse_args()


def main():
    ## Load argument variables
    args = args_parser()

    if os.path.exists(args.input) is False:
        raise Exception("Input Folder Path Do Not Exists")

    if os.path.exists(args.output) is False:
        raise Exception("Output Folder Path Do Not Exists")

    ## Retrieve trained model & label map through Datature Hub
    hub_model = HubModel(
        project_secret=args.project_secret,
        model_key=args.model_key,
    )

    print("Loading model...")
    start_time = time.time()
    trained_model = hub_model.load_tf_model()
    category_index = hub_model.load_label_map()
    print("Model loaded, took {} seconds...".format(time.time() - start_time))

    all_detections = []
    all_images = glob.glob(os.path.join(args.input, "*"))

    ## Run prediction on each image
    for each_image in all_images:
        print("Predicting for {}...".format(each_image))

        input_tensor = hub_model.load_image_with_model_dimensions(each_image)
        detections_output = trained_model(input_tensor)
        all_detections.append(detections_output)

    ## Save predicted image
    for each_image, each_detection in zip(all_images, all_detections):
        visualized_image = visualize_bbox(
            each_image, each_detection, category_index, args.threshold
        )
        filename = os.path.basename(each_image)
        image_predict = Image.fromarray(visualized_image)
        image_predict.save(os.path.join(args.output, filename))

        print(
            "Saving predicted images to {}...".format(
                os.path.join(args.output, filename)
            )
        )
