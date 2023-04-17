import argparse
import glob
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from utils.postprocess import postprocess


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
    output_name = session.get_outputs()[0].name
    ## Run prediction on each image
    for each_image in glob.glob(os.path.join(args.input, "*")):
        print("Predicting for {}...".format(each_image))

        ## Returned original_shape is in the format of width, height
        image_resized, origi_shape = load_image_into_numpy_array(
            each_image, int(height), int(width))
        input_image = np.expand_dims(
            image_resized.astype(np.float32).transpose(2, 0, 1), 0)

        ## Feed image into model
        start_time = time.time()
        detections_output = session.run([output_name],
                                        {input_name: input_image})
        end_time = time.time()
        print("Inference time: ", start_time - end_time)
        print(detections_output[0].shape)
        output = postprocess(detections_output[0], num_classes=3)[0]
        bboxes = output[:, 0:4].cpu().detach().numpy().astype(np.float64)
        classes = output[:, 6].cpu().detach().numpy().astype(np.float64)
        scores = (output[:, 4] * output[:, 5]).cpu().detach().numpy().astype(
            np.float64)
        bboxes = [[
            bbox[1] / height,
            bbox[0] / width,
            bbox[3] / height,
            bbox[2] / width,
        ] for bbox in bboxes]

        ## Draw Predictions
        image_origi = Image.fromarray(image_resized).resize(
            (origi_shape[1], origi_shape[0]))
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
                        str(category_index[classes[idx] + 1]["name"]),
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

            print("Saving predicted images to {}...".format(
                os.path.join(args.output, filename)))


if __name__ == "__main__":
    main()
