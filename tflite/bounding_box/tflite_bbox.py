### Import all dependencies
import os
import cv2
import glob
import numpy as np
import tensorflow as tf

from PIL import Image
from matplotlib.pyplot import imshow
import argparse


## Prepare utility functions
def load_label_map(label_map_path):
    """Reads label map in the format of .pbtxt and parse into dictionary

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

    return label_map


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
        "--size", help="Size of image to load into model", default="320x320"
    )
    parser.add_argument(
        "--threshold", help="Prediction confidence threshold", default=0.7
    )
    parser.add_argument(
        "--model", help="Path to tensorflowlite model", default="./tf.lite"
    )
    parser.add_argument(
        "--label",
        help="Path to tensorflow label map",
        default="./label_map.pbtxt",
    )
    return parser.parse_args()


def main():
    ## Load argument variables
    args = args_parser()

    height, width = args.size.split("x")

    if os.path.exists(args.input) is False:
        raise Exception("Input Folder Path Do Not Exists")

    if os.path.exists(args.output) is False:
        raise Exception("Output Folder Path Do Not Exists")

    ### Load label map and color map
    category_index = load_label_map(args.label)

    color_map = {}
    for each_class in range(len(category_index)):
        color_map[each_class] = [
            int(i) for i in np.random.choice(range(256), size=3)
        ]

    ## Load model into interpreter
    interpreter = tf.lite.Interpreter(model_path=args.model)

    ## Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    ## Redefine expected image size of input layer
    interpreter.resize_tensor_input(
        input_details[0]["index"], (1, int(width), int(height), 3)
    )
    interpreter.allocate_tensors()

    ## Run prediction on each image
    for each_image in glob.glob(os.path.join(args.input, "*")):
        print("Predicting for {}...".format(each_image))

        ## Load image into numpy array and resize accordingly
        img = cv2.imread(each_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origi_shape = img.shape
        new_img = cv2.resize(img, (int(width), int(height)))

        ## Execute prediction
        interpreter.set_tensor(
            input_details[0]["index"], [new_img.astype(np.uint8)]
        )
        interpreter.invoke()

        ## Extract data and filter against desired threshold
        for each_layer in output_details:
            if each_layer["name"] == "StatefulPartitionedCall:4":
                scores = np.squeeze(
                    interpreter.get_tensor(each_layer["index"])
                )

            if each_layer["name"] == "StatefulPartitionedCall:1":
                bboxes = np.squeeze(
                    interpreter.get_tensor(each_layer["index"])
                )

            if each_layer["name"] == "StatefulPartitionedCall:2":
                classes = np.squeeze(
                    interpreter.get_tensor(each_layer["index"])
                )

        ## Filter out predictions below threshold
        indexes = np.where(scores > float(args.threshold))

        ## Filter all prediction data above threshold
        scores = scores[indexes]
        bboxes = bboxes[indexes]
        classes = classes[indexes]

        ## Draw mask, bounding box, label onto the original image
        image_origi = np.array(img)

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
