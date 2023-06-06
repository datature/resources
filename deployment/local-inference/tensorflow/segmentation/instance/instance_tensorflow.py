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
@Desc    :   Simple predictor script for instance segmentation TF models.
'''

import argparse
import os
from typing import Dict, List

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

HEIGHT, WIDTH = NotImplemented, NotImplemented


def label_map(label_path: str) -> Dict[str, str]:
    """Process the label map file."""
    return_label_map = {}
    with open(label_path, "r") as label_file:
        for line in label_file:
            if "id" in line:
                label_index = int(line.split(":")[-1])
                label_name = next(label_file).split(":")[-1].strip().strip(
                    "'\"")
                return_label_map[int(label_index)] = label_name
    return_label_map[0] = "Background"
    return return_label_map


def get_instance_mask(msk: np.ndarray, lab: int, thresh: int) -> np.ndarray:
    """Convert class mask to instaance mask"""
    instance_mask = np.zeros_like(msk, np.uint8)
    instance_mask[np.where(msk > thresh)] = lab + 1
    return instance_mask


class Predictor:

    def __init__(self, model, height, width, out_name, thresh):
        self.model = model
        self.model_height = height
        self.model_width = width
        self.output_name = out_name
        self.threshold = thresh

    def feed_forward(self, instance: List[np.ndarray]) -> Dict:
        """Single feed-forward pipeline"""
        result = self.model(inputs=instance)
        return result

    def preprocess(self, input_image: np.ndarray) -> np.ndarray:
        """Preprocess the image array before it is fed to the model"""
        model_input = Image.fromarray(input_image).convert("RGB")
        model_input = model_input.resize((self.model_width, self.model_height))
        model_input = np.array(model_input).astype(np.float32)
        model_input = np.expand_dims(model_input, 0)
        return model_input

    def postprocess(self, model_output: List[np.ndarray]) -> List:
        """Postprocess the model output"""
        sco = np.array(model_output["output_1"][0])
        classes = np.array(model_output["output_2"][0]).astype(np.int16)
        boxes = np.array(model_output["output_3"][0])
        masks = np.array(model_output["output_4"][0])
        _filter = np.where(sco > self.threshold)
        sco = sco[_filter]
        classes = classes[_filter]
        boxes = boxes[_filter]
        masks = masks[_filter]

        masks_output = []
        for cls, each_mask in zip(classes, masks):
            output_mask = get_instance_mask(each_mask, cls, self.threshold)
            masks_output.append(output_mask)

        if masks_output:
            masks_output = np.stack(masks_output)
        else:
            masks_output = None

        return [boxes, masks_output, sco, classes]

    def predict(self, image: np.ndarray) -> Dict:
        """Send an image for prediction, then process it to be returned
            afterwards.
        """
        preprocessed = self.preprocess(image)
        predicted = self.feed_forward(preprocessed)
        postprocessed = self.postprocess(predicted)
        return postprocessed


parser = argparse.ArgumentParser(
    prog="Datature Instance Segmentation Tensorflow Predictor",
    description="Predictor to Predict Instance Segmentation Tensorflow Model.")
parser.add_argument("-i", "--input_folder_path")
parser.add_argument("-o", "--output_folder_path")
parser.add_argument("-m", "--model_path")
parser.add_argument("-l", "--label_map_path")
parser.add_argument("-t", "--threshold")

if __name__ == "__main__":
    args = parser.parse_args()
    input_path = args.input_folder_path
    output_path = args.output_folder_path
    model_path = args.model_path
    label_map_path = args.label_map_path
    threshold = float(args.threshold)

    loaded = tf.saved_model.load(model_path)
    loaded_model = loaded.signatures["serving_default"]
    output_name = list(loaded_model.structured_outputs.keys())[0]
    predictor = Predictor(loaded_model, HEIGHT, WIDTH, output_name, threshold)

    color_map = {}
    num_classes = 3
    for each_class in range(1, num_classes):
        color_map[each_class] = [
            int(i) for i in np.random.choice(range(256), size=3)
        ]
    category_map = label_map(label_map_path)

    for image_name in os.listdir(input_path):
        if ".jpg" not in image_name and ".png" not in image_name:
            print(
                f"Only .jpg and .png files can be used, {image_name} skipped")
            continue
        print("\nPredicting for", image_name)

        img = Image.open(os.path.join(input_path, image_name)).convert("RGB")
        img_array = np.array(img)
        bboxes, output_masks, scores, labels = predictor.predict(img_array)

        if len(bboxes) != 0 and output_masks is not None:
            for each_bbox, label, score in zip(bboxes, labels, scores):
                label += 1
                color = color_map.get(label)

                ## Draw bounding box
                cv2.rectangle(
                    img_array,
                    (
                        int(each_bbox[1] * img_array.shape[0]),
                        int(each_bbox[0] * img_array.shape[1]),
                    ),
                    (
                        int(each_bbox[3] * img_array.shape[0]),
                        int(each_bbox[2] * img_array.shape[1]),
                    ),
                    color,
                    2,
                )

                ## Draw label background
                cv2.rectangle(
                    img_array,
                    (
                        int(each_bbox[1] * img_array.shape[0]),
                        int(each_bbox[2] * img_array.shape[1]),
                    ),
                    (
                        int(each_bbox[3] * img_array.shape[0]),
                        int(each_bbox[2] * img_array.shape[1] + 15),
                    ),
                    color,
                    -1,
                )

                ## Insert label class & score
                cv2.putText(
                    img_array,
                    "Class: {}, Score: {}".format(
                        str(category_map[label]),
                        str(round(score, 2)),
                    ),
                    (
                        int(each_bbox[1] * img_array.shape[0]),
                        int(each_bbox[2] * img_array.shape[1] + 10),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            for mask, each_bbox in zip(output_masks, bboxes):
                mask = np.expand_dims(mask, -1)
                mask = np.tile(mask, 3)
                mask *= 255
                mask = np.clip(mask, 0, 255).astype(np.uint8)
                xmin, ymin, xmax, ymax = each_bbox
                ymin = int(ymin * img_array.shape[1])
                ymax = int(ymax * img_array.shape[1])
                xmin = int(xmin * img_array.shape[0])
                xmax = int(xmax * img_array.shape[0])
                mask_height = ymax - ymin
                mask_width = xmax - xmin
                mask = cv2.resize(mask, (mask_height, mask_width))
                resized_mask = np.zeros(img_array.shape)
                resized_mask[xmin:xmax, ymin:ymax, :] = mask
                img_array = img_array.astype(np.int64)
                img_array += resized_mask.astype(np.int64)
                img_array = np.clip(img_array, 0, 255)

            img_mask = Image.fromarray(img_array.astype(np.uint8))
            specific_output_path = os.path.join(output_path, image_name)
            img_mask.save(specific_output_path)
            print("Prediction saved to", specific_output_path)

        else:
            print("No detections for", image_name)
