#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
  ████
██    ██   Datature
  ██  ██   Powering Breakthrough AI
    ██

@File    :   predict.py
@Author  :   Leonard So
@Version :   1.0
@Contact :   hello@datature.io
@License :   Apache License 2.0
@Desc    :   Simple predictor script for semantic segmentation TF models
             with area calculation overlaid on top of the saved image.
'''

import argparse
import os
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

HEIGHT, WIDTH = 1024, 1024
PIXEL_LENGTH = 205
TRUE_LENGTH = 20


def calculate_true_area(pixel_area: float, pixel_length: int,
                        true_length: float) -> float:
    """
    Args:
        pixel_area: This is the area of the object based on pixels
        as a unit area.
        pixel_length: Pixel length is the length in number of pixels
        for the true_length.
        true_length: The real world length equivalent in pixels.
    Returns:
        true_area: This is the real world area for the pixel area.
    """
    true_area = (pixel_area / (pixel_length**2)) * true_length**2
    return true_area


def draw_area(mask: np.ndarray, origi_image: Image.Image) -> Image.Image:
    """Used to annotate masks with area overlaying the image."""
    image = origi_image
    draw = ImageDraw.Draw(image)
    contours = cv2.findContours(
        (mask * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_TC89_KCOS,
    )[0]
    for each_contour in contours:
        contour_array = np.array(each_contour)
        if contour_array.shape.count(1):
            contour_array = np.squeeze(contour_array)

        ## If polygon has less than three vertices then skip
        if contour_array.size < 3 * 2:
            continue
        M = cv2.moments(each_contour)
        area = cv2.contourArea(each_contour)
        true_area = calculate_true_area(area, PIXEL_LENGTH, TRUE_LENGTH)
        if true_area < 10:
            continue
        fontsize = 50
        font = ImageFont.truetype("./arial.ttf", fontsize)
        if M["m00"] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            draw.text(
                (cx - font.getsize(f"area: {int(true_area)} sq-ft")[0] / 2,
                 cy),
                f"area: {int(true_area)} sq-ft",
                font=font)
    return image


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


def get_binary_mask(mask: np.ndarray, threshold: float) -> np.ndarray:
    """Convert class mask to binary mask"""
    binary_mask = np.zeros_like(mask[0], np.uint8)
    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    for class_id, class_mask in enumerate(mask):
        if class_id > 0:
            binary_mask[np.where(class_mask > threshold)] = class_id
    return binary_mask


class Predictor:

    def __init__(self, model, height, width, output_name, threshold):
        self.model = model
        self.model_height = height
        self.model_width = width
        self.output_name = output_name
        self.threshold = threshold

    def preprocess(self, input_image: np.ndarray) -> np.ndarray:
        """Preprocess the image array before it is fed to the model"""
        model_input = Image.fromarray(input_image).convert("RGB")
        model_input = model_input.resize((self.model_width, self.model_height))
        model_input = np.array(model_input).astype(np.float32)
        model_input = np.expand_dims(model_input, 0)
        return model_input

    def postprocess(self, model_output: List[np.ndarray]) -> np.ndarray:
        """Postprocess the model output"""
        class_masks = model_output[self.output_name]
        semantic_masks = get_binary_mask(class_masks[0], self.threshold)
        return semantic_masks

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Send an image for prediction, then process it to be returned
            afterwards.
        """
        preprocessed = self.preprocess(image)
        predicted = self.model(inputs=preprocessed)
        postprocessed = self.postprocess(predicted)
        return postprocessed


parser = argparse.ArgumentParser(
    prog="Datature Semantic Segmentation TF Predictor",
    description="Predictor to Predict Semantic Segmentation TF Saved Model.")
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
    loaded_output_name = list(loaded_model.structured_outputs.keys())[0]
    predictor = Predictor(loaded_model, HEIGHT, WIDTH, loaded_output_name,
                          threshold)

    category_map = label_map(label_map_path)
    color_map = {}
    num_classes = len(category_map)
    for each_class in range(1, num_classes):
        color_map[each_class] = np.array(
            [int(i) for i in np.random.choice(range(256), size=3)],
            dtype=np.uint8)
    color_map[0] = np.array([0, 0, 0], dtype=np.uint8)

    for image_name in os.listdir(input_path):
        if ".jpg" not in image_name and ".png" not in image_name:
            print(
                f"Only .jpg and .png files can be used, {image_name} skipped")
            continue
        img = Image.open(os.path.join(input_path, image_name)).convert("RGB")
        img_array = np.array(img)

        original_mask = predictor.predict(img_array)
        if original_mask is not None:
            predicted_mask = np.expand_dims(original_mask, -1)
            predicted_mask = np.tile(predicted_mask, 3)
            unique = np.unique(predicted_mask)
            output_mask = np.zeros_like(predicted_mask, dtype=np.uint8)
            class_mask_list = []
            for num, item in enumerate(unique):
                output_mask = np.where(predicted_mask != item, output_mask,
                                       color_map[item])
                binary_class_mask = cv2.resize(
                    np.where(original_mask == item, 255, 0).astype(np.uint8),
                    (img_array.shape[1], img_array.shape[0]))
                class_mask_list.append(binary_class_mask)

            output_mask = cv2.resize(output_mask,
                                     (img_array.shape[1], img_array.shape[0]))
            img_array = img_array.astype(np.uint8)
            pil_img = Image.fromarray(img_array)
            pil_mask = Image.fromarray(output_mask)
            pil_img = Image.blend(pil_img, pil_mask, 0.7)
            for mask in class_mask_list:
                pil_img = draw_area(mask, pil_img)
            specific_output_path = os.path.join(output_path, image_name)
            mask_path = os.path.join(output_path, f"mask_{image_name}")
            pil_img.save(specific_output_path)
            pil_mask.save(mask_path)
            print("Composite Image saved to", specific_output_path)
            print("Mask saved to", mask_path)
        else:
            print("No detections for", image_name)
