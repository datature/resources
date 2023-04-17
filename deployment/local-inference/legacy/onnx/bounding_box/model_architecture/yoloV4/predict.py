import argparse
import glob
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from yolo_utils.postprocess import yolov3v4_postprocess
from utils.postprocess import postprocess


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
                label_map[label_index] = {
                    "id": label_index,
                    "name": label_name,
                }

    label_map[0] = {"id": 0, "name": '"Background"'}
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

    image_resized = image.resize((width, height))
    return np.array(image_resized), (image_shape[0], image_shape[1])


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
        help="Path to tensorflow pb model",
        required=True,
    )
    parser.add_argument(
        "--label",
        help="Path to tensorflow label map",
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

    ## Load label map
    category_index = load_label_map(args.label)

    ## Load color map
    color_map = {}
    for each_class in range(len(category_index)):
        color_map[each_class] = [
            int(i) for i in np.random.choice(range(256), size=3)
        ]

    ## Load model
    print("Loading model...")
    start_time = time.time()
    session = ort.InferenceSession(
        args.model,
        providers=["CUDAExecutionProvider"],
    )
    print("Model loaded, took {} seconds...".format(time.time() - start_time))

    input_name = session.get_inputs()[0].name
    output_names = [
        single_output.name for single_output in session.get_outputs()
    ]
    anchors = get_anchors("yolo_utils/yolo4_anchors.txt")
    ## Run prediction on each image
    for each_image in glob.glob(os.path.join(args.input, "*")):
        print("Predicting for {}...".format(each_image))

        ## Returned original_shape is in the format of width, height
        image_resized, origi_shape = load_image_into_numpy_array(
            each_image, int(height), int(width))

        ## Feed image into model
        start_time = time.time()
        detections_output = session.run(
            output_names, {
                input_name:
                np.expand_dims(
                    (image_resized / 255).astype(np.uint8), axis=0)
            })
        end_time = time.time()
        # detections_output = [np.squeeze(detection)for detection in detections_output]
        print("Inference time: ", start_time - end_time)
        bboxes, classes, scores = yolov3v4_postprocess(
            detections_output,
            (int(width), int(height)),
            anchors,
            len(category_index),
            (int(width), int(height)),
        )
        
        indexes = np.where(scores > float(args.threshold))
        
        bboxes = bboxes[indexes]
        classes = classes[indexes]
        scores = scores[indexes]
        
        ## Draw Predictions
        image_origi = np.array(
            Image.fromarray(image_resized.astype(np.uint8)).resize(
                (origi_shape[1], origi_shape[0])))

        if len(bboxes) != 0:
            for idx, each_bbox in enumerate(bboxes):

                color = color_map.get(classes[idx])

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
