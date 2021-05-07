"""
Prediction script for trained tensorflow PB model.
command to run script
python3 prediction.py --input ./input --output ./output --size 640x640 --model . --label ./label_map.pbtxt

"""
import os
import time
import glob
import argparse
import numpy as np
import tensorflow as tf
import cv2
from dataturehub.hub import hub
from dataturehub.visualize import visualize
from PIL import Image

# Comment out next line to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Comment out to set verbose to true
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def args_parser():
    parser = argparse.ArgumentParser(
        description="Datature Open Source Prediction Script"
    )
    parser.add_argument(
        "--input",
        help="Path to folder that contains input images",
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Path to folder to store predicted images",
        required=True,
    )
    parser.add_argument(
        "--size", help="Size of image to load into model", default=None
    )
    parser.add_argument(
        "--threshold", help="Prediction confidence threshold", default=0.7
    )
    parser.add_argument(
        "--model", help="Path to tensorflow pb model", default=None
    )
    parser.add_argument(
        "--label", help="Path to tensorflow label map", default=None
    )
    return parser.parse_args()


def main():
    # Load argument variables
    args = args_parser()

    if os.path.exists(args.input) is False:
        raise Exception("Input Folder Path Do Not Exists")

    if os.path.exists(args.output) is False:
        raise Exception("Output Folder Path Do Not Exists")

    if args.size is not None:
        height, width = args.size.split("x")

    hubb = hub()
    trained_model = hubb.load_tf_model(model_dir=args.model)
    category_index = hubb.load_label_map_from_file(label_map_path=args.label)
    # Run prediction on each image
    for each_image in glob.glob(os.path.join(args.input, "*")):
        print("Predicting for {}...".format(each_image))
        input_tensor = hubb.load_image(each_image, int(height), int(width))
        # Feed image into model
        detections_output = trained_model(input_tensor)

        visualized_image = visualize(
            each_image, detections_output, category_index, args.threshold
        ).visualize_bbox()

        # Save predicted image
        filename = os.path.basename(each_image)
        image_predict = Image.fromarray(visualized_image)
        image_predict.save(os.path.join(args.output, filename))

        print(
            "Saving predicted images to {}...".format(
                os.path.join(args.output, filename)
            )
        )


if __name__ == "__main__":
    main()
