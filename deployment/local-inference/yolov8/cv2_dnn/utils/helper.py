#!/usr/env/bin python
# -*-coding:utf-8 -*-
"""
  ████
██    ██   Datature
  ██  ██   Powering Breakthrough AI
    ██

@File    :   helper.py
@Author  :   Wei Loon Cheng
@Version :   1.0
@Contact :   hello@datature.io
@License :   Apache License 2.0
@Desc    :   Helper functions for preprocessing and postprocessing.
"""

import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


def load_label_map(label_map_path: str) -> Dict:
    """
    Reads label map in the format of .pbtxt and parse into dictionary

    Args:
        label_map_path: the file path to the label_map

    Returns:
        dictionary with the format of
        {
            label_index: {
                'id': label_index,
                'name': label_name
            }
        }
    """
    label_map = {}

    with open(label_map_path, "r", encoding="utf-8") as label_file:
        for line in label_file:
            if "id" in line:
                label_index = int(line.split(":")[-1])
                label_name = next(label_file).split(":")[-1].strip().strip("'")
                label_map[label_index] = {
                    "id": label_index,
                    "name": label_name.strip('"'),
                }

    return label_map


def draw_predictions(
    image: np.ndarray,
    results: np.ndarray,
    category_index: Dict[int, str],
    color_map: Dict[int, list],
):
    """Draws bounding boxes and labels on prediction results.

    Args:
        image (np.ndarray): Original image.
        results (np.ndarray): Prediction results.
        category_index (Dict[int, str]): Dictionary of category index and name.
        color_map (Dict[int, list]): Dictionary of category index and color.

    Returns:
        np.ndarray: Image with bounding boxes and labels.
    """
    bboxes = results[0][:, :4]
    classes = results[0][:, 5]
    scores = results[0][:, 4]

    for each_bbox, each_class, each_score in zip(bboxes, classes, scores):
        color = color_map.get(each_class - 1)

        # Draw bbox on screen
        cv2.rectangle(
            image,
            (int(each_bbox[0]), int(each_bbox[1])),
            (int(each_bbox[2]), int(each_bbox[3])),
            color,
            2,
        )
        # Draw label background
        cv2.rectangle(
            image,
            (int(each_bbox[0]), int(each_bbox[3])),
            (int(each_bbox[2]), int(each_bbox[3] + 15)),
            color,
            thickness=-1,
        )
        cv2.putText(
            image,
            f"{category_index[int(each_class)]['name']}, "
            f"Score: {str(round(each_score, 2))}",
            (int(each_bbox[0]), int(each_bbox[3] + 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return image


def letterbox_resize(
    image: np.ndarray,
    target_size: Tuple[int, int],
    fill_color: Tuple[int, int, int] = (114, 114, 114),
):
    """
    Resize an image with letterboxing.

    Parameters:
        image (np.ndarray): The input image.
        target_size (tuple[int, int]): The target size of the output image (width, height).
        fill_color (tuple[int, int, int]): The color to use for the letterboxing.

    Returns:
        (np.ndarray) Resized image with letterboxing.
    """

    # Calculate the resizing scale
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)

    # Calculate the new size after resizing
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image
    resized_img = cv2.resize(image, (new_w, new_h))

    # Create a blank canvas with the target size and fill with the specified color
    canvas = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)

    # Calculate the position to paste the resized image with letterboxing
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Paste the resized image onto the canvas
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized_img

    return canvas


def postprocess(
    preds: np.ndarray,
    model_shape: Tuple[int, int],
    img_shape: Tuple[int, int],
    conf: float,
):
    """Post-processes predictions and returns a list of Results objects."""
    preds = non_max_suppression(preds, conf_thres=conf)
    results = []
    for pred in preds:
        pred[:, :4] = scale_boxes(model_shape, pred[:, :4], img_shape)
        results.append(pred)
    return results


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.7,
    classes: Optional[List[int]] = None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels: List[List[Union[int, float, np.ndarray]]] = (),
    max_det: int = 300,
    nc: int = 0,
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks
    and multiple labels per box.

    Args:
        prediction (np.ndarray):
            A numpy array of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The array should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]):
            A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, np.ndarray]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this
            will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes for non-max suppression.
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[np.ndarray]): A list of length batch_size, where each element is a numpy array of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = np.max(prediction[:, 4:mi], 1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = np.transpose(
        prediction, (0, 2, 1)
    )  # shape(1, 84, 6300) to shape(1, 6300, 84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy

    t = time.time()
    output = [np.zeros((0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = np.zeros((len(lb), nc + nm + 4))
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = np.split(x, (4, 4 + nc), axis=1)

        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate(
                (box[i], x[i, 4 + j, None], j[:, None].astype(float), mask[i]), axis=1
            )
        else:
            j = np.argmax(cls, axis=1)[:, np.newaxis]
            conf = np.max(cls, axis=1)[:, np.newaxis]
            x = np.concatenate((box, conf, j.astype(float), mask), axis=1)[
                conf.flatten() > conf_thres
            ]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[
                x[:, 4].argsort(descending=True)[:max_nms]
            ]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def xywh2xyxy(box: np.ndarray):
    """
    Convert bounding box coordinates from (x, y, width, height) format to
    (x1, y1, x2, y2) format where (x1, y1) is the top-left corner and
    (x2, y2) is the bottom-right corner.

    Args:
        box (np.ndarray): The input bounding box coordinates
            in (x, y, width, height) format.

    Returns:
        new_box (np.ndarray): The bounding box coordinates in
            (x1, y1, x2, y2) format.
    """
    assert (
        box.shape[-1] == 4
    ), f"input shape last dimension expected 4 but input shape is {box.shape}"
    new_box = np.empty_like(box)
    dw = box[..., 2] / 2  # half-width
    dh = box[..., 3] / 2  # half-height
    new_box[..., 0] = box[..., 0] - dw  # top left x
    new_box[..., 1] = box[..., 1] - dh  # top left y
    new_box[..., 2] = box[..., 0] + dw  # bottom right x
    new_box[..., 3] = box[..., 1] + dh  # bottom right y
    return new_box


def nms(boxes: np.ndarray, scores: np.ndarray, thresh: float):
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than threshold with another (higher scoring) box.

    Args:
        boxes (np.ndarray): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (np.ndarray): scores for each one of the boxes
        thresh (float): discards all overlapping boxes with IoU > thresh

    Returns:
        np.ndarray: int64 numpy array with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        inter = w * h
        iou = inter / (
            np.maximum(0.0, (boxes[i, 2] - boxes[i, 0]))
            * np.maximum(0.0, (boxes[i, 3] - boxes[i, 1]))
            + (boxes[order[1:], 2] - boxes[order[1:], 0])
            * np.maximum(0.0, (boxes[order[1:], 3] - boxes[order[1:], 1]))
            - inter
        )

        inds = np.where(iou <= thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def scale_boxes(
    img1_shape: Tuple[int, int],
    boxes: np.ndarray,
    img0_shape: Tuple[int, int],
    ratio_pad: Tuple[float, float] = None,
    padding: bool = True,
):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image
    they were originally specified in (img1_shape) to the shape of a different
    image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are
            for, in theformat of (height, width).
        boxes (np.ndarray): the bounding boxes of the objects in the image,
            in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of
            (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes.
            If not provided, the ratio and pad will be calculated based on the
            size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented
            by yolo style. If False then do regular rescaling.

    Returns:
        boxes (np.ndarray): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., [0, 2]] -= pad[0]  # x padding
        boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    """
    Takes a list of bounding boxes and a shape (height, width) and clips the
    bounding boxes to the shape.

    Args:
      boxes (np.ndarray): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
