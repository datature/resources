#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
  ████
██    ██   Datature
  ██  ██   Powering Breakthrough AI
    ██

@File    :   predict.py
@Author  :   Marcus Neo
@Version :   1.0
@Contact :   hello@datature.io
@License :   Apache License 2.0
@Desc    :   Simple predictor script for YOLOv8 models.
'''
import argparse

from ultralytics import YOLO

## Change this to your model input size
HEIGHT, WIDTH = NotImplemented, NotImplemented


def predict(model, input_folder, threshold):
    model = YOLO(model)
    model.predict(
        source=input_folder,
        conf=threshold,
        imgsz=[WIDTH, HEIGHT],
        save=True,
        task='detect',
    )


parser = argparse.ArgumentParser(
    prog="Datature-Ultralytics YOLOV8 Predictor",
    description="Predictor to Predict with Datature-Ultralytics YOLOV8 Model.")

parser.add_argument("-i", "--input_folder_path")
parser.add_argument("-m", "--model_path")
parser.add_argument("-t", "--threshold")


def main():
    args = parser.parse_args()
    input_path = args.input_folder_path
    model_path = args.model_path
    threshold = float(args.threshold)

    predict(
        model=model_path,
        input_folder=input_path,
        threshold=threshold,
    )


if __name__ == "__main__":
    main()
