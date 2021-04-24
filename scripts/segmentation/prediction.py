"""
Prediction script for trained tensorflow PB model.
"""
import os
import time
import glob
import argparse
import numpy as np
import tensorflow as tf
import cv2

from PIL import Image


## Comment out next line to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
## Comment out to set verbose to true
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def reframe_box_masks_to_image_masks(
    box_masks, boxes, image_height, image_width, resize_method="bilinear"
):
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
        unit_boxes = tf.concat([tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], 1)
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


def load_label_map(label_map_path):
    """Reads label map in the format of .pbtxt and parse into dictionary

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
                label_map[label_index] = {"id": label_index, "name": label_name}

    return label_map


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


def load_image_into_numpy_array(path, height, width):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = Image.open(path).convert("RGB")
    image_shape = np.asarray(image).shape

    image_resized = image.resize((height, width))
    return np.array(image_resized), (image_shape[0], image_shape[1])


def args_parser():
    parser = argparse.ArgumentParser(
        description="Blade Defect Detection Prediction Service"
    )
    parser.add_argument(
        "--input", help="Path to folder that contains input images", required=True
    )
    parser.add_argument(
        "--output", help="Path to folder to store predicted images", required=True
    )
    parser.add_argument(
        "--size", help="Size of image to load into model", default="1024x1024"
    )
    parser.add_argument(
        "--threshold", help="Prediction confidence threshold", default=0.7
    )
    parser.add_argument(
        "--model", help="Path to tensorflow pb model", default="./saved_model"
    )
    parser.add_argument(
        "--label", help="Path to tensorflow label map", default="./label_map.pbtxt"
    )
    return parser.parse_args()


def main():
    ## Load argument variables
    args = args_parser()

    if os.path.exists(args.input) is False:
        raise Exception("Input Folder Path Do Not Exists")

    if os.path.exists(args.output) is False:
        raise Exception("Output Folder Path Do Not Exists")

    if os.path.exists(args.model) is False:
        raise Exception("Model Folder Do Not Exists")

    category_index = load_label_map(args.label)

    ## Load color map
    color_map = {}
    for each_class in range(len(category_index)):
        color_map[each_class] = [int(i) for i in np.random.choice(range(256), size=3)]

    ## Load model
    print("Loading model...")
    start_time = time.time()
    detect_fn = tf.saved_model.load(args.model)
    print("Model loaded, took {} seconds...".format(time.time() - start_time))

    ## Run prediction on each image
    for each_image in glob.glob(os.path.join(args.input, "*")):
        print("Prediction for {}...".format(each_image))

        height, width = args.size.split("x")

        ## Returned original_shape is in the format of width, height
        image_resized, origi_shape = load_image_into_numpy_array(
            each_image, int(height), int(width)
        )

        ## The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_resized)

        ## The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        ## Feed image into model
        trained_model = detect_fn.signatures["serving_default"]
        detections = trained_model(input_tensor)

        ## Process predictions
        num_detections = int(detections.pop("num_detections"))

        need_detection_key = [
            "detection_classes",
            "detection_boxes",
            "detection_masks",
            "detection_scores",
        ]

        predictions = {
            key: detections[key][0, :num_detections].numpy()
            for key in need_detection_key
        }

        ## Filter out predictions below threshold
        predictions["num_detections"] = num_detections
        indexes = np.where(predictions["detection_scores"] > float(args.threshold))

        if "detection_masks" in predictions:
            # Reframe the the bbox mask to the image size.
            detection_masks_reframed = reframe_box_masks_to_image_masks(
                tf.convert_to_tensor(predictions["detection_masks"]),
                predictions["detection_boxes"],
                origi_shape[0],
                origi_shape[1],
            )
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
            predictions["detection_masks_reframed"] = detection_masks_reframed.numpy()

        ## Extract predictions
        masks = predictions["detection_masks_reframed"][indexes]
        bboxes = predictions["detection_boxes"][indexes]
        classes = predictions["detection_classes"][indexes].astype(np.int64)
        scores = predictions["detection_scores"][indexes]

        ## Draw Predictions
        image_origi = Image.fromarray(image_resized).resize(
            (origi_shape[1], origi_shape[0])
        )
        image_origi = np.array(image_origi)

        if len(masks) != 0:
            for idx, each_bbox in enumerate(bboxes):
                color = color_map.get(classes[idx] - 1)
                masked_image = apply_mask(image_origi, masks[idx], color)

                ## Draw bounding box
                cv2.rectangle(
                    masked_image,
                    (
                        int(each_bbox[1] * origi_shape[1]),
                        int(each_bbox[0] * origi_shape[0]),
                    ),
                    (
                        int(each_bbox[3] * origi_shape[1]),
                        int(each_bbox[2] * origi_shape[0]),
                    ),
                    color,
                    2,
                )

                ## Draw label background
                cv2.rectangle(
                    masked_image,
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
                    masked_image,
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
            filename = os.path.basename(each_image)
            image_predict = Image.fromarray(masked_image)
            image_predict.save(os.path.join(args.output, filename))

            print(
                "Saving predicted images to {}...".format(
                    os.path.join(args.output, filename)
                )
            )


if __name__ == "__main__":
    main()
