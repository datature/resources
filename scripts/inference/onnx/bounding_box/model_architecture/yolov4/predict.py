"""
Prediction script for trained onnx .onnx yolo model.
"""

import os
import cv2
import time
import glob
import argparse
import numpy as np
import absl.logging
import onnxruntime as ort

from PIL import Image
from yolo_utils.postprocess import yolov3v4_postprocess

## Disable unnecessary warnings
absl.logging.set_verbosity(absl.logging.ERROR)
## Comment out next line to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_anchors(anchors_path):
    """loads the yolo anchors from a text file

    Args:
        anchors_path: the file path to anchors text file

    Returns:
        array of loaded 2d floats
    """
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(",")]

    return np.array(anchors).reshape(-1, 2)


def load_label_map(label_map_path):
    """
    Reads label map in the format of .txt and parse into dictionary

    Args:
      label_map_path: the file path to the label_map

    Returns:
      dictionary with the format of {label_index: {'id': label_index, 'name': label_name}}
    """
    label_map = {}

    with open(label_map_path, "r") as label_file:
        for idx, line in enumerate(label_file):
            tag_name = line.rstrip("\n")
            label_map[idx] = {
                "id": idx,
                "name": tag_name,
            }

    return label_map


def load_image_into_numpy_array(path, width, height):
    """
    Load an image from file into a numpy array.

    Args:
      path: the file path to the image
      width: width of image
      height: height of image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3), (original_height, original_width)
    """
    image = Image.open(path).convert("RGB")
    image_shape = np.asarray(image).shape

    image_resized = image.resize((width, height))
    return np.array(image_resized).astype("float32"), (
        image_shape[0],
        image_shape[1],
    )


def args_parser():
    parser = argparse.ArgumentParser(
        description="Datature Open Source Prediction Script")
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
        "--model",
        help="Path to onnx model",
        required=True,
    )
    parser.add_argument(
        "--label",
        help="Path to yolo label map",
        required=True,
    )
    parser.add_argument(
        "--anchors",
        help="Yolo model anchors text file",
        required=True,
    )
    parser.add_argument("--width",
                        help="Width of image to load into model",
                        default=640)
    parser.add_argument("--height",
                        help="Height of image to load into model",
                        default=640)
    parser.add_argument("--threshold",
                        help="Prediction confidence threshold",
                        default=0.7)

    return parser.parse_args()


def main():
    ## Load argument variables
    args = args_parser()

    width = int(args.width)
    height = int(args.height)

    if os.path.exists(args.input) is False:
        raise Exception("Input Folder Path Do Not Exists")

    if os.path.exists(args.output) is False:
        raise Exception("Output Folder Path Do Not Exists")

    if os.path.exists(args.model) is False:
        raise Exception("Model Folder Do Not Exists")

    if os.path.exists(args.label) is False:
        raise Exception("Label Map Do Not Exists")

    if os.path.exists(args.anchors) is False:
        raise Exception("Anchors File Do Not Exists")

    ## Load label map
    category_index = load_label_map(args.label)

    ## Load color map
    color_map = {}
    for each_class in range(len(category_index)):
        color_map[each_class] = [
            int(i) for i in np.random.choice(range(256), size=3)
        ]

    ## Load anchors
    anchors = get_anchors(args.anchors)

    ## Load model
    print("Loading model...")
    start_time = time.time()
    session = ort.InferenceSession(
        args.model,
        providers=["CUDAExecutionProvider"],
    )

    ## Save Input Output layer names
    input_name = session.get_inputs()[0].name

    output_names = [
        single_output.name for single_output in session.get_outputs()
    ]
    print("Model loaded, took {} seconds...".format(time.time() - start_time))

    ## Run prediction on each image
    for each_image in glob.glob(os.path.join(args.input, "*")):
        print("Predicting for {}...".format(each_image))

        ## Returned original_shape is in the format of width, height
        image_resized, origi_shape = load_image_into_numpy_array(
            each_image, int(height), int(width))

        ## Normalize input image
        input_image = (image_resized / 255.0).astype(np.float32)

        detections_output = session.run(
            output_names, {input_name: np.expand_dims(input_image, axis=0)})

        bboxes, classes, scores = yolov3v4_postprocess(
            detections_output, (int(width), int(height)),
            anchors,
            3, (int(width), int(height)),
            confidence=args.threshold)

        ## Draw Predictions
        image_origi = np.array(
            Image.fromarray(image_resized.astype(np.uint8)).resize(
                (origi_shape[1], origi_shape[0])))

        if len(bboxes) != 0:
            for idx, each_bbox in enumerate(bboxes):

                color = color_map.get(classes[idx] - 1)

                ## Draw bounding box
                cv2.rectangle(
                    image_origi,
                    (
                        int(each_bbox[0] * origi_shape[1]),
                        int(each_bbox[1] * origi_shape[0]),
                    ),
                    (
                        int(each_bbox[2] * origi_shape[1]),
                        int(each_bbox[3] * origi_shape[0]),
                    ),
                    color,
                    2,
                )

                # Draw label background
                cv2.rectangle(
                    image_origi,
                    (
                        int(each_bbox[0] * origi_shape[1]),
                        int(each_bbox[3] * origi_shape[0]),
                    ),
                    (
                        int(each_bbox[2] * origi_shape[1]),
                        int(each_bbox[3] * origi_shape[0] + 15),
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
                        int(each_bbox[0] * origi_shape[1]),
                        int(each_bbox[3] * origi_shape[0] + 10),
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

            print("Saving predicted images to {}...".format(
                os.path.join(args.output, filename)))


if __name__ == "__main__":
    main()