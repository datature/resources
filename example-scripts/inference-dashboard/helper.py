#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
  ████
██    ██   Datature
  ██  ██   Powering Breakthrough AI
    ██

@File    :   helper.py
@Author  :   Wei Loon Cheng
@Version :   1.0
@Contact :   hello@datature.io
@License :   Apache License 2.0
@Desc    :   Datature inference dashboard helper functions.
'''

import copy
import os

import numpy as np
import tensorflow as tf


def load_label_map(label_path):
    """Read label map in the format of .txt and parse into dictionary

    Args:
        label_map_path: the file path to the label_map

    Returns:
        dictionary with the format of
            {label_index: {'id': label_index, 'name': label_name}}
    """

    if os.path.exists(label_path) is False:
        raise FileNotFoundError("No valid label map found.")

    label_map = {}

    with open(label_path, "r", encoding="utf-8") as label_file:
        for line in label_file:
            # If labels file is in JSON format
            if "id" in line:
                label_index = int(line.split(":")[-1])
                label_name = next(label_file).split(":")[-1].strip().strip("'")
                label_map[label_index] = {
                    "id": label_index,
                    "name": label_name,
                }

    label_map[0] = {"id": 0, "name": '"background"'}
    return label_map


def reframe_box_masks_to_image_masks(box_masks,
                                     boxes,
                                     image_height,
                                     image_width,
                                     resize_method="bilinear"):
    """Transforms the box masks back to full image masks.

    Embeds masks in bounding boxes of larger masks whose shapes correspond to
    image shape.

    Args:
      box_masks: A tensor of size [num_masks, mask_height, mask_width].
      boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
             corners. Row i contains [ymin, xmin, ymax, xmax] of the box
             corresponding to mask i. Note that the box corners are in
             normalized coordinates.
      image_height: Image height. The output mask will have the same height as
                    the image height.
      image_width: Image width. The output mask will have the same width as the
                   image width.
      resize_method: The resize method, either 'bilinear' or 'nearest'. Note that
        'bilinear' is only respected if box_masks is a float.

    Returns:
      A tensor of size [num_masks, image_height, image_width] with the same dtype
      as `box_masks`.
    """
    resize_method = "nearest" if box_masks.dtype == tf.uint8 else resize_method

    def reframe_box_masks_to_image_masks_default():
        """The default function when there are more than 0 box masks."""

        def transform_boxes_relative_to_boxes(boxes, reference_boxes):
            boxes = tf.reshape(boxes, [-1, 2, 2])
            min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
            max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
            denom = max_corner - min_corner
            # Prevent a divide by zero.
            denom = tf.math.maximum(denom, 1e-4)
            transformed_boxes = (boxes - min_corner) / denom
            return tf.reshape(transformed_boxes, [-1, 4])

        box_masks_expanded = tf.expand_dims(box_masks, axis=3)
        num_boxes = tf.shape(box_masks_expanded)[0]
        unit_boxes = tf.concat(
            [tf.zeros([num_boxes, 2]),
             tf.ones([num_boxes, 2])], 1)
        reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)

        resized_crops = tf.image.crop_and_resize(
            box_masks_expanded,
            reverse_boxes,
            tf.range(num_boxes),
            [image_height, image_width],
            method=resize_method,
            extrapolation_value=0,
        )
        return tf.cast(resized_crops, box_masks.dtype)

    image_masks = tf.cond(
        tf.shape(box_masks)[0] > 0,
        reframe_box_masks_to_image_masks_default,
        lambda: tf.zeros([0, image_height, image_width, 1], box_masks.dtype),
    )
    return tf.squeeze(image_masks, axis=3)


def apply_mask(image, mask, colors, alpha=0.5):
    """Apply the given mask to the image.
    Args:
      image: original image array.
      mask: predict mask array of image.
      colors: color to apply for mask.
      alpha: transparency of mask.
    Returns:
      array of image with mask overlay
    """
    for color in range(3):
        image[:, :, color] = np.where(
            mask == 1,
            image[:, :, color] * (1 - alpha) + alpha * colors[color],
            image[:, :, color],
        )
    return image


def load_image_into_numpy_array(image, height, width):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    image_shape = np.asarray(image).shape
    image_resized = image.resize((width, height))
    return np.array(image_resized), (image_shape[0], image_shape[1])


def nms_boxes(
    boxes,
    classes,
    scores,
    iou_threshold,
    confidence=0.1,
    sigma=0.5,
):
    """Carry out non-max supression on the detected bboxes"""
    use_diou = True
    is_soft = False
    use_exp = False

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
