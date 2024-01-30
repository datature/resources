#!/usr/env/bin python
# -*-coding:utf-8 -*-
"""
  ████
██    ██   Datature
  ██  ██   Powering Breakthrough AI
    ██

@File    :   bbox_val.py
@Author  :   Tze Lynn Kho
@Version :   1.0
@Contact :   hello@datature.io
@License :   Apache License 2.0
@Desc    :   Evaluate performance of YOLOv8 object detection
             models on a validation set.
"""

import argparse
import json
from typing import Dict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


def load_cls_mapping(val_anno_path: str, label_map_path: str) -> Dict:
    """
    Maps the class index from the prediction to the class index in the validation set.
    This is needed because the class index in the prediction may not be the same as the
    class index in the validation set.

    Args:
        val_anno_path (str): the file path to the validation annotation file
        label_map_path (str): the file path to the label map

    Returns:
        dictionary with the format of {pred_cls_index: val_cls_index}
    """
    cls_mapping = {}
    with open(val_anno_path, "r", encoding="utf-8") as f:
        val_anns = json.load(f)
        val_cls = {}
        for cat in val_anns["categories"]:
            val_cls[cat["name"]] = cat["id"]

    with open(label_map_path, "r", encoding="utf-8") as label_file:
        for line in label_file:
            if "id" in line:
                label_index = int(line.split(":")[-1])
                label_name = next(label_file).split(":")[-1].strip().strip('"')
                if label_name in val_cls.keys():
                    cls_mapping[label_index] = val_cls[label_name]
    return cls_mapping


def load_img_id_mapping(val_anno_path: str) -> Dict:
    """Loads the mapping from image file name to image id in the validation set.

    Args:
        val_anno_path (str): the file path to the validation annotation file

    Returns:
        Dict: dictionary with the format of {img_file_name: img_id}
    """
    img_id_mapping = {}
    with open(val_anno_path, "r", encoding="utf-8") as f:
        val_anns = json.load(f)
        for img in val_anns["images"]:
            img_id_mapping[img["file_name"]] = img["id"]
    return img_id_mapping


parser = argparse.ArgumentParser(
    description="Evaluate performance of YOLOv8 object"
    "detection models on a validation set."
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
    help="Path to YOLOv8 object detection model",
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
parser.add_argument(
    "--save",
    action="store_true",
    help="Save the predicted images",
)


def main():
    args = parser.parse_args()
    model = YOLO(args.model_path, task="detect")

    preds = model.predict(
        source=args.val_img_path,
        conf=args.threshold,
        imgsz=[args.input_size, args.input_size],
        save=args.save,
        verbose=False,
    )

    cls_mapping = load_cls_mapping(args.val_anno_path, args.label_path)
    img_id_mapping = load_img_id_mapping(args.val_anno_path)

    dets = []
    for pred in preds:
        img_id = pred.path.split("/")[-1]
        for p in pred:
            bbox = p.boxes.xyxy.tolist()[0]
            bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
            dct = {
                "image_id": img_id_mapping[img_id],
                "category_id": cls_mapping[int(p.boxes.cls.item())],
                "bbox": bbox,
                "score": p.boxes.conf.item(),
            }
            dets.append(dct)

    coco_annotation = COCO(annotation_file=args.val_anno_path)
    coco_eval_prediction = coco_annotation.loadRes(dets)
    coco_eval = COCOeval(coco_annotation, coco_eval_prediction, "bbox")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    main()
