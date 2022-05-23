"""
Prediction script for trained tensorflow PB model.
"""

import os
import time
import glob
import argparse
import numpy as np
import tensorflow as tf
import cv2

from PIL import Image

## Comment out next line to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
## Comment out to set verbose to true
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_label_map(label_map_path):
    """
    Reads label map in the format of .pbtxt and parse into dictionary

    Args:
      label_map_path: the file path to the label_map

    Returns:
      dictionary with the format of {label_index: {'id': label_index, 'name': label_name}}
    """
    label_map = {}

    with open(label_map_path, "r") as label_file:
        for line in label_file:
            if "id" in line:
                label_index = int(line.split(":")[-1])
                label_name = next(label_file).split(":")[-1].strip().strip("'")
                label_map[label_index] = {"id": label_index, "name": label_name}

    return label_map


def load_image_into_numpy_array(path, height, width):
    """
    Load an image from file into a numpy array.

    Args:
      path: the file path to the image
      height: height of image
      width: width of image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3), (original_height, original_width)
    """
    image = Image.open(path).convert("RGB")
    image_shape = np.asarray(image).shape

    image_resized = image.resize((height, width))
    return np.array(image_resized), (image_shape[0], image_shape[1])


def args_parser():
    parser = argparse.ArgumentParser(
        description="Datature Open Source Prediction Script"
    )
    parser.add_argument(
        "--input", help="Path to folder that contains input images", required=True
    )
    parser.add_argument(
        "--output", help="Path to folder to store predicted images", required=True
    )
    parser.add_argument(
        "--size", help="Size of image to load into model", default="320x320"
    )
    parser.add_argument(
        "--threshold", help="Prediction confidence threshold", default=0.7
    )
    parser.add_argument(
        "--model", help="Path to tensorflow pb model", default="./saved_model"
    )
    parser.add_argument(
        "--label", help="Path to tensorflow label map", default="./label_map.pbtxt"
    )
    return parser.parse_args()


def main():
    ## Load argument variables
    args = args_parser()

    if os.path.exists(args.input) is False:
        raise Exception("Input Folder Path Do Not Exists")

    if os.path.exists(args.output) is False:
        raise Exception("Output Folder Path Do Not Exists")

    if os.path.exists(args.model) is False:
        raise Exception("Model Folder Do Not Exists")

    category_index = load_label_map(args.label)

    ## Load color map
    color_map = {}
    for each_class in range(len(category_index)):
        color_map[each_class] = [int(i) for i in np.random.choice(range(256), size=3)]

    ## Load model
    print("Loading model...")
    start_time = time.time()
    trained_model = tf.saved_model.load(args.model)
    print("Model loaded, took {} seconds...".format(time.time() - start_time))

    ## Run prediction on each image
    for each_image in glob.glob(os.path.join(args.input, "*")):
        print("Predicting for {}...".format(each_image))

        height, width = args.size.split("x")

        ## Returned original_shape is in the format of width, height
        image_resized, origi_shape = load_image_into_numpy_array(
            each_image, int(height), int(width)
        )

        ## The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_resized)

        ## The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        ## Feed image into model
        detections_output = trained_model(input_tensor)

        num_detections = int(detections_output.pop("num_detections"))
        detections = {
            key: value[0, :num_detections].numpy()
            for key, value in detections_output.items()
        }
        detections["num_detections"] = num_detections

        ## Filter out predictions below threshold
        indexes = np.where(detections["detection_scores"] > float(args.threshold))

        ## Extract predictions
        bboxes = detections["detection_boxes"][indexes]
        classes = detections["detection_classes"][indexes].astype(np.int64)
        scores = detections["detection_scores"][indexes]

        ## Draw Predictions
        image_origi = Image.fromarray(image_resized).resize(
            (origi_shape[1], origi_shape[0])
        )
        image_origi = np.array(image_origi)

        if len(bboxes) != 0:
            for idx, each_bbox in enumerate(bboxes):
                color = color_map.get(classes[idx] - 1)

                ## Draw bounding box
                cv2.rectangle(
                    image_origi,
                    (
                        int(each_bbox[1] * origi_shape[1]),
                        int(each_bbox[0] * origi_shape[0]),
                    ),
                    (
                        int(each_bbox[3] * origi_shape[1]),
                        int(each_bbox[2] * origi_shape[0]),
                    ),
                    color,
                    2,
                )

                ## Draw label background
                cv2.rectangle(
                    image_origi,
                    (
                        int(each_bbox[1] * origi_shape[1]),
                        int(each_bbox[2] * origi_shape[0]),
                    ),
                    (
                        int(each_bbox[3] * origi_shape[1]),
                        int(each_bbox[2] * origi_shape[0] + 15),
                    ),
                    color,
                    -1,
                )

                ## Insert label class & score
                cv2.putText(
                    image_origi,
                    "Class: {}, Score: {}".format(
                        str(category_index[classes[idx]]["name"]),
                        str(round(scores[idx], 2)),
                    ),
                    (
                        int(each_bbox[1] * origi_shape[1]),
                        int(each_bbox[2] * origi_shape[0] + 10),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            ## Save predicted image
            filename = os.path.basename(each_image)
            image_predict = Image.fromarray(image_origi)
            image_predict.save(os.path.join(args.output, filename))

            print(
                "Saving predicted images to {}...".format(
                    os.path.join(args.output, filename)
                )
            )


if __name__ == "__main__":
    main()
