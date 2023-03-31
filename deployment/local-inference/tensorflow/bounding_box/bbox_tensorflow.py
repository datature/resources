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
@Desc    :   Simple predictor script for bounding box Tensorflow models.
'''

import argparse
import copy
import glob
import os
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

HEIGHT, WIDTH = NotImplemented, NotImplemented


def load_label_map(label_map_path: str) -> Dict:
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


def load_image_into_numpy_array(path: str, height: int, width: int) -> List:
    """
    Load an image from file into a numpy array.

    Args:
      path: the file path to the image
      height: height of image for model input
      width: width of image for model input

    Returns:
      List containing:
        1. uint8 numpy array with shape (img_height, img_width, 3)
        2. Tuple of (original_height, original_width) of image
    """
    image = Image.open(path).convert("RGB")
    image_shape = np.asarray(image).shape
    image_resized = image.resize((width, height))
    return [np.array(image_resized), (image_shape[0], image_shape[1])]


def nms_boxes(
    boxes,
    classes,
    scores,
    iou_threshold,
    confidence=0.1,
    sigma=0.5,
):
    """Carry out non-max supression on the detected bboxes"""
    use_diou=True
    is_soft=False
    use_exp=False

    nboxes, nclasses, nscores = [], [], []
    for cls in set(classes):
        # handle data for one class
        inds = np.where(classes == cls)
        bbx = boxes[inds]
        cls = classes[inds]
        sco = scores[inds]

        # make a data copy to avoid breaking
        # during nms operation
        b_nms = copy.deepcopy(bbx)
        c_nms = copy.deepcopy(cls)
        s_nms = copy.deepcopy(sco)

        while len(s_nms) > 0:
            # pick the max box and store, here
            # we also use copy to persist result
            i = np.argmax(s_nms, axis=-1)
            nboxes.append(copy.deepcopy(b_nms[i]))
            nclasses.append(copy.deepcopy(c_nms[i]))
            nscores.append(copy.deepcopy(s_nms[i]))

            # swap the max line and first line
            b_nms[[i, 0], :] = b_nms[[0, i], :]
            c_nms[[i, 0]] = c_nms[[0, i]]
            s_nms[[i, 0]] = s_nms[[0, i]]

            iou = box_diou(b_nms)
                
            # drop the last line since it has been record
            b_nms = b_nms[1:]
            c_nms = c_nms[1:]
            s_nms = s_nms[1:]

            if is_soft:
                # Soft-NMS
                if use_exp:
                    # score refresh formula:
                    # score = score * exp(-(iou^2)/sigma)
                    s_nms = s_nms * np.exp(-(iou * iou) / sigma)
                else:
                    # score refresh formula:
                    # score = score * (1 - iou) if iou > threshold
                    depress_mask = np.where(iou > iou_threshold)[0]
                    s_nms[depress_mask] = s_nms[depress_mask] * (
                        1 - iou[depress_mask])
                keep_mask = np.where(s_nms >= confidence)[0]
            else:
                # normal Hard-NMS
                keep_mask = np.where(iou <= iou_threshold)[0]

            # keep needed box for next loop
            b_nms = b_nms[keep_mask]
            c_nms = c_nms[keep_mask]
            s_nms = s_nms[keep_mask]

    # reformat result for output
    nboxes = np.array(nboxes)
    nclasses = np.array(nclasses)
    nscores = np.array(nscores)
    return nboxes, nclasses, nscores


def box_diou(boxes):
    """
    Calculate DIoU value of 1st box with other boxes of a box array
    Reference Paper:
        "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
        https://arxiv.org/abs/1911.08287

    Args:
      boxes: bbox numpy array, shape=(N, 4), xywh
             x,y are top left coordinates

    Returns:
      diou: numpy array, shape=(N-1,)
            IoU value of boxes[1:] with boxes[0]
    """
    # get box coordinate and area
    x_pos = boxes[:, 0]
    y_pos = boxes[:, 1]
    wid = boxes[:, 2]
    hei = boxes[:, 3]
    areas = wid * hei

    # check IoU
    inter_xmin = np.maximum(x_pos[1:], x_pos[0])
    inter_ymin = np.maximum(y_pos[1:], y_pos[0])
    inter_xmax = np.minimum(x_pos[1:] + wid[1:], x_pos[0] + wid[0])
    inter_ymax = np.minimum(y_pos[1:] + hei[1:], y_pos[0] + hei[0])

    inter_w = np.maximum(0.0, inter_xmax - inter_xmin + 1)
    inter_h = np.maximum(0.0, inter_ymax - inter_ymin + 1)

    inter = inter_w * inter_h
    iou = inter / (areas[1:] + areas[0] - inter)

    # box center distance
    x_center = x_pos + wid / 2
    y_center = y_pos + hei / 2
    center_distance = np.power(x_center[1:] - x_center[0], 2) + np.power(
        y_center[1:] - y_center[0], 2)

    # get enclosed area
    enclose_xmin = np.minimum(x_pos[1:], x_pos[0])
    enclose_ymin = np.minimum(y_pos[1:], y_pos[0])
    enclose_xmax = np.maximum(x_pos[1:] + wid[1:], x_pos[0] + wid[0])
    enclose_ymax = np.maximum(y_pos[1:] + wid[1:], y_pos[0] + wid[0])
    enclose_w = np.maximum(0.0, enclose_xmax - enclose_xmin + 1)
    enclose_h = np.maximum(0.0, enclose_ymax - enclose_ymin + 1)
    # get enclosed diagonal distance
    enclose_diagonal = np.power(enclose_w, 2) + np.power(enclose_h, 2)
    # calculate DIoU, add epsilon in denominator to avoid dividing by 0
    diou = iou - 1.0 * (center_distance) / (enclose_diagonal +
                                            np.finfo(float).eps)

    return diou


parser = argparse.ArgumentParser(
    prog="Datature Bounding Box Tensorflow Predictor",
    description="Predictor to Predict Bounding Box Tensorflow Model.")
parser.add_argument("-i", "--input_folder_path")
parser.add_argument("-o", "--output_folder_path")
parser.add_argument("-m", "--model_path")
parser.add_argument("-l", "--label_map_path")
parser.add_argument("-t", "--threshold")


def main():
    args = parser.parse_args()
    input_path = args.input_folder_path
    output_path = args.output_folder_path
    model_path = args.model_path
    label_map_path = args.label_map_path
    threshold = float(args.threshold)

    ## Load label map
    category_index = load_label_map(label_map_path)

    ## Load color map
    color_map = {}
    for each_class in range(len(category_index)):
        color_map[each_class] = [
            int(i) for i in np.random.choice(range(256), size=3)
        ]

    ## Load model
    loaded = tf.saved_model.load(model_path)
    model = loaded.signatures["serving_default"]
    output_name = list(model.structured_outputs.keys())[0]

    ## Run prediction on each image
    for each_image in glob.glob(os.path.join(input_path, "*")):
        if ".jpg" not in each_image and ".png" not in each_image:
            print(
                f"Only .jpg and .png files can be used, {each_image} skipped")
            continue
        print("Predicting for {}...".format(each_image))

        ## Resize image and
        ## return original image shape in the format (width, height)
        image_resized, origi_shape = load_image_into_numpy_array(
            each_image, int(HEIGHT), int(WIDTH))
        input_image = np.expand_dims(image_resized.astype(np.float32), 0)

        ## Feed image into model
        detections_output = model(inputs=input_image)

        ## Filter detections
        detections_output = np.array(detections_output[output_name][0])
        slicer = detections_output[:, -1]
        output = detections_output[:, :6][slicer != 0]
        scores = output[:, 4]
        output = output[scores > threshold]
        classes = output[:, 5]
        output = output[classes != 0]

        ## Postprocess detections
        bboxes = output[:, :4]
        classes = output[:, 5].astype(np.int32)
        scores = output[:, 4]
        bboxes[:, 0], bboxes[:,
                             1] = (bboxes[:, 1] * WIDTH, bboxes[:, 0] * HEIGHT)
        bboxes[:, 2], bboxes[:,
                             3] = (bboxes[:, 3] * WIDTH, bboxes[:, 2] * HEIGHT)
        bboxes, classes, scores = nms_boxes(bboxes, classes, scores, 0.1)
        bboxes = [[
            bbox[1] / WIDTH,
            bbox[0] / HEIGHT,
            bbox[3] / WIDTH,
            bbox[2] / HEIGHT,
        ] for bbox in bboxes]  # y1, x1, y2, x2

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
                        int(each_bbox[1] * origi_shape[1]),  # x1
                        int(each_bbox[0] * origi_shape[0]),  # y1
                    ),
                    (
                        int(each_bbox[3] * origi_shape[1]),  # x2
                        int(each_bbox[2] * origi_shape[0]),  # y2
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
            each_image = os.path.basename(each_image)
            image_predict = Image.fromarray(image_origi)
            specific_output_path = os.path.join(output_path, each_image)
            image_predict.save(specific_output_path)
            print("Prediction saved to", specific_output_path)
        else:
            print("No detections for", each_image)


if __name__ == "__main__":
    main()
