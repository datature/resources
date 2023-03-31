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
@Desc    :   Simple predictor script for semantic segmentation ONNX models.
'''

import argparse
import os
from typing import List

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

HEIGHT, WIDTH = NotImplemented, NotImplemented


def get_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class mask to binary mask"""
    binary_mask = np.zeros_like(mask[0], np.uint8)
    for class_id, class_mask in enumerate(mask):
        if class_id > 0:
            binary_mask[np.where(class_mask > 0.0)] = class_id
    return binary_mask


class Predictor:

    def __init__(self, model):
        self.model = model
        self.model_inputs = self.model.get_inputs()
        self.model_input_names = list(map(lambda x: x.name, self.model_inputs))
        self.model_outputs = self.model.get_outputs()
        self.model_output_names = list(
            map(lambda x: x.name, self.model_outputs))
        model_shape = self.model_inputs[0].shape
        self.model_height = model_shape[1]
        self.model_width = model_shape[2]
        self.num_classes = model_shape[3]

    def feed_forward(self, instance: List[np.ndarray]) -> List[np.ndarray]:
        """Single Onnx feed-forward pipeline"""
        result = self.model.run(self.model_output_names,
                                dict(zip(self.model_input_names, instance)))
        return result

    def preprocess(self, input_image: np.ndarray) -> List[np.ndarray]:
        """Preprocess the image array before it is fed to the model"""
        model_input = Image.fromarray(input_image).convert("RGB")
        model_input = model_input.resize((self.model_width, self.model_height))
        model_input = np.array(model_input).astype(np.float32)
        model_input = np.expand_dims(model_input, 0)
        return [model_input]

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Send an image for prediction, then process it to be returned
            afterwards.
        """
        preprocessed = self.preprocess(image)
        predicted = self.feed_forward(preprocessed)[0][0]
        postprocessed = get_binary_mask(predicted)
        return postprocessed


parser = argparse.ArgumentParser(
    prog="Datature Semantic Segmentation ONNX Predictor",
    description="Predictor to Predict Semantic Segmentation ONNX Model.")
parser.add_argument("-i", "--input_folder_path")
parser.add_argument("-o", "--output_folder_path")
parser.add_argument("-m", "--model_path")

if __name__ == "__main__":
    args = parser.parse_args()
    input_path = args.input_folder_path
    output_path = args.output_folder_path
    model_path = args.model_path
    loaded_model = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider"],  # or "CPUExecutionProvider"
    )
    predictor = Predictor(loaded_model)

    for image_name in os.listdir(input_path):
        if ".jpg" not in image_name and ".png" not in image_name:
            print(
                f"Only .jpg and .png files can be used, {image_name} skipped")
            continue
        img = Image.open(os.path.join(input_path, image_name)).convert("RGB")
        img_array = np.array(img)

        output_mask = predictor.predict(img_array)
        if output_mask is not None:
            output_mask = np.expand_dims(output_mask, -1)
            output_mask = np.tile(output_mask, 3)
            output_mask *= 127
            output_mask = np.clip(output_mask, 0, 255).astype(np.uint8)
            output_mask = cv2.resize(output_mask,
                                     (img_array.shape[1], img_array.shape[0]))
            img_array = img_array.astype(np.int64)
            img_array += output_mask
            img_array = np.clip(img_array, 0, 255)
            img_mask = Image.fromarray(img_array.astype(np.uint8))
            specific_output_path = os.path.join(output_path, image_name)
            img_mask.save(specific_output_path)
            print("Prediction saved to", specific_output_path)
        else:
            print("No detections for", image_name)
