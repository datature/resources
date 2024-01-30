#!/usr/env/bin python
# -*-coding:utf-8 -*-
"""
  ████
██    ██   Datature
  ██  ██   Powering Breakthrough AI
    ██

@File    :   cls_val.py
@Author  :   Tze Lynn Kho
@Version :   1.0
@Contact :   hello@datature.io
@License :   Apache License 2.0
@Desc    :   Evaluate performance of YOLOv8 classification
             models on a validation set.
"""

import argparse
import os

from sklearn.metrics import accuracy_score, f1_score
from ultralytics import YOLO

parser = argparse.ArgumentParser(
    description="Evaluate performance of YOLOv8 classifcation"
    "models on a validation set."
)
parser.add_argument(
    "-i",
    "--val_img_path",
    type=str,
    required=True,
    help="Path to folder containing validation images",
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    required=True,
    help="Path to YOLOv8 classification model",
)
parser.add_argument(
    "-a",
    "--val_anno_path",
    type=str,
    required=True,
    help="Path to validation annotation file",
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


def main():
    args = parser.parse_args()
    model = YOLO(args.model_path, task="classify")

    preds = model.predict(
        source=args.val_img_path,
        conf=args.threshold,
        imgsz=[args.input_size, args.input_size],
        save=False,
        verbose=False,
    )

    pred_labels = []
    pred_scores = []
    for pred in preds:
        pred_labels.append(pred.probs.top1)
        pred_scores.append(pred.probs.data.tolist())

    img_names = os.listdir(args.val_img_path)
    img_names.sort()

    label_map = {}
    with open(args.label_path, "r", encoding="utf-8") as label_file:
        for line in label_file:
            if "id" in line:
                label_index = int(line.split(":")[-1])
                label_name = next(label_file).split(":")[-1].strip().strip("'")
                label_name = label_name.split("_")[0].replace('"', "")
                label_map[label_name] = label_index
    gt_labels = []
    for img_name in img_names:
        gt_labels.append(label_map[img_name.split("_")[0]])

    f1_scores = f1_score(gt_labels, pred_labels, average="macro")
    acc_scores = accuracy_score(gt_labels, pred_labels)

    print(f"F1 Score: {f1_scores:.2%}")
    print(f"Accuracy Score: {acc_scores:.2%}")


if __name__ == "__main__":
    main()
