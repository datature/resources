#!/usr/env/bin python
# -*-coding:utf-8 -*-
"""
  ████
██    ██   Datature
  ██  ██   Powering Breakthrough AI
    ██

@File    :   predict.py
@Author  :   Wei Loon Cheng
@Version :   1.0
@Contact :   hello@datature.io
@License :   Apache License 2.0
@Desc    :   Datature prediction script for YOLOv8 ONNX models
             using OpenCV DNN
"""

import argparse
import os

import cv2
import numpy as np
from utils.helper import draw_predictions, letterbox_resize, load_label_map, postprocess

parser = argparse.ArgumentParser(
    description="Datature prediction script for YOLOv8 ONNX models using OpenCV DNN"
)
parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    required=True,
    help="Path to folder containing images",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    required=True,
    help="Path to output folder",
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    required=True,
    help="Path to ONNX model",
)
parser.add_argument(
    "-s",
    "--input_size",
    type=int,
    required=True,
    help="Input size of the model, e.g. 320 or 640",
)
parser.add_argument(
    "-l",
    "--label_path",
    type=str,
    required=True,
    help="Path to label map",
)
parser.add_argument(
    "-t",
    "--threshold",
    type=float,
    default=0.5,
    help="Prediction confidence threshold",
)


def predict(image, net, input_size):
    image = letterbox_resize(image, (input_size, input_size))
    blob = cv2.dnn.blobFromImage(
        image=image,
        scalefactor=1.0 / 255,
        size=(input_size, input_size),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    preds = net.forward()
    return preds


def main():
    args = parser.parse_args()

    net = cv2.dnn.readNetFromONNX(args.model_path)

    category_index = load_label_map(args.label_path)
    color_map = {}
    for each_class in range(len(category_index)):
        color_map[each_class] = [int(i) for i in np.random.choice(range(256), size=3)]

    for file_name in os.listdir(args.input_dir):
        image_path = os.path.join(args.input_dir, file_name)
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        origi_image = image.copy()

        preds = predict(image, net, args.input_size)

        results = postprocess(
            preds=preds,
            model_shape=(args.input_size, args.input_size),
            img_shape=(height, width),
            conf=args.threshold,
        )
        origi_image = draw_predictions(origi_image, results, category_index, color_map)
        output_path = os.path.join(args.output_dir, file_name)
        cv2.imwrite(output_path, origi_image)


if __name__ == "__main__":
    main()
