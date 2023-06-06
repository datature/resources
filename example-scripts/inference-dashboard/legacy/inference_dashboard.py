#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
  ████
██    ██   Datature
  ██  ██   Powering Breakthrough AI
    ██
@File    :   inference_dashboard.py
@Author  :   Wei Loon Cheng
@Version :   1.0
@Contact :   hello@datature.io
@License :   Apache License 2.0
@Desc    :   Datature inference dashboard built using Streamlit for prediction visualisations (legacy).
'''

import json
import os
from io import BytesIO

import absl.logging
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from datature_hub.hub import HubModel
from helper import (
    apply_mask,
    load_image_into_numpy_array,
    load_label_map,
    reframe_box_masks_to_image_masks,
)
from PIL import Image

## Disable unnecessary warnings for tensorflow
absl.logging.set_verbosity(absl.logging.ERROR)
## Comment out next line to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
## Comment out to set verbose to true
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_local_model():
    """Load model locally using model path and label map path."""
    if st.session_state.local_hash != st.session_state.model_path + st.session_state.label_map_path:
        with st.spinner('Loading model...'):
            ## Load Tensorflow model
            st.session_state.trained_model = tf.saved_model.load(
                st.session_state.model_path)

            ## Load label map
            st.session_state.category_index = load_label_map(
                st.session_state.label_map_path)

            ## Load color map
            for each_class in range(len(st.session_state["category_index"])):
                st.session_state.color_map = {}
                st.session_state.color_map[each_class] = [
                    int(i) for i in np.random.choice(range(256), size=3)
                ]
        st.session_state.local_hash = st.session_state.model_path + st.session_state.label_map_path
        st.success('Model loaded successfully!')


def load_hub_model():
    """Load model from Datature Hub using model key and project secret."""
    if st.session_state.hub_hash != st.session_state.project_secret + st.session_state.model_key:
        with st.spinner('Loading model...'):
            ## Initialise Datature Hub
            st.session_state.hub: HubModel = HubModel(
                model_key=st.session_state.model_key,
                project_secret=st.session_state.project_secret,
            )

            ## Load Tensorflow model
            st.session_state.trained_model = st.session_state.hub.load_tf_model(
            ).signatures["serving_default"]

            ## Load label map
            st.session_state.category_index = st.session_state.hub.load_label_map(
            )

            ## Load color map
            for each_class in range(len(st.session_state["category_index"])):
                st.session_state.color_map = {}
                st.session_state.color_map[each_class] = [
                    int(i) for i in np.random.choice(range(256), size=3)
                ]
        st.session_state.hub_hash = st.session_state.project_secret + st.session_state.model_key
        st.success('Model loaded successfully!')


def upload_images():
    """Upload images function for the sidebar."""
    ## Upload images
    st.sidebar.file_uploader(
        "Image uploader",
        type=['jpg', 'png', 'jpeg'],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload image(s) to predict. Supported file types: jpg, png, jpeg",
        key="uploaded_imgs",
    )

    st.session_state.file_names = [
        each_file.name for each_file in st.session_state.uploaded_imgs
    ]


def predict(img_path):
    """Predict function for each image.
    Args:
        img_path: path to the image file
    Returns:
        a tuple of the original image as a numpy array,
        shape of the original image, and model predictions
    """
    ## Load argument variables
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    image_resized, origi_shape = load_image_into_numpy_array(
        image, int(height), int(width))

    ## The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_resized)

    ## The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    ## Feed image into model
    detections_output = st.session_state.trained_model(input_tensor)

    num_detections = int(detections_output.pop("num_detections"))
    detections = {
        key: value[0, :num_detections].numpy()
        for key, value in detections_output.items()
    }
    detections["num_detections"] = num_detections

    return np.array(image), origi_shape, detections


def predict_all():
    """Predict all the uploaded images and store the predictions in the session state."""
    with st.spinner('Running predictions...'):
        for each_file in st.session_state.uploaded_imgs:
            if each_file.name not in st.session_state.prediction_cache.keys():
                origi_image, origi_shape, detections = predict(each_file)

                st.session_state.prediction_cache.update({
                    each_file.name: {
                        "img": origi_image,
                        "shape": origi_shape,
                        "detections": detections
                    }
                })


def draw_detections(origi_image, origi_shape, detections):
    """Apply the given mask to the image and draw the predictions on the image.
    Args:
        origi_image: original image
        origi_shape: shape of the original image (img_height, img_width)
        detections: model predictions
    Returns:
        output image with predictions drawn on it
    """
    if "detection_masks_reframed" in detections:
        ## Extract predictions
        masks = detections["detection_masks_reframed"]

    for idx, (each_bbox, each_class, each_score) in enumerate(
            zip(detections["detection_boxes"], detections["detection_classes"],
                detections["detection_scores"])):
        color = st.session_state.color_map.get(each_class - 1)
        if "detection_masks_reframed" in detections:
            origi_image = apply_mask(origi_image, masks[idx], color)
        ## Draw bounding box
        cv2.rectangle(
            origi_image,
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
            origi_image,
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
            origi_image,
            "Class: {}, Score: {}".format(
                str(st.session_state.category_index[each_class]["name"]),
                str(round(each_score, 2)),
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

    return Image.fromarray(origi_image, mode="RGB")


def get_json_output(detections):
    """Generates JSON output of the model predictions.
    Args:
        detections: model predictions
    Returns:
        dictionary of the model predictions
    """
    json_output = {"predictions": []}

    for idx, (each_bbox, each_class, each_score) in enumerate(
            zip(detections["detection_boxes"], detections["detection_classes"],
                detections["detection_scores"])):
        ## [y_min, x_min, y_max, x_max]
        xmin = float(each_bbox[1])
        ymin = float(each_bbox[0])
        xmax = float(each_bbox[3])
        ymax = float(each_bbox[2])

        bound = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]

        prediction = {
            "annotationId": idx,
            "bound": bound,
            "boundType": "rectangle",
            "confidence": float(each_score),
            "contour": None,
            "contourType": None,
            "tag": {
                "id": int(each_class),
                "name":
                str(st.session_state.category_index[each_class]["name"])
            }
        }

        json_output["predictions"].append(prediction)

    return json_output


def visualise():
    """Perform visualisation by filtering out predictions with scores below a threshold,
    drawing the thresholded predictions on the image and generating JSON output."""
    for each_file_name in st.session_state.prediction_cache.keys():
        if each_file_name in st.session_state.file_names:
            st.slider(f"Threshold_{each_file_name}",
                      min_value=0.0,
                      max_value=1.0,
                      value=0.7,
                      step=0.01,
                      label_visibility="collapsed",
                      help="Threshold",
                      key=f"threshold_{each_file_name}")

            origi_image = st.session_state.prediction_cache[each_file_name][
                "img"]
            origi_shape = st.session_state.prediction_cache[each_file_name][
                "shape"]
            detections = st.session_state.prediction_cache[each_file_name][
                "detections"]

            threshold_detections = {}
            indexes = np.where(detections["detection_scores"] > float(
                st.session_state[f"threshold_{each_file_name}"]))
            threshold_detections["detection_boxes"] = detections[
                "detection_boxes"][indexes]
            threshold_detections["detection_classes"] = detections[
                "detection_classes"][indexes].astype(np.int64)
            threshold_detections["detection_scores"] = detections[
                "detection_scores"][indexes]

            if "detection_masks" in detections:
                ## Reframe the the bbox mask to the image size.
                detection_masks_reframed = reframe_box_masks_to_image_masks(
                    tf.convert_to_tensor(detections["detection_masks"]),
                    detections["detection_boxes"],
                    origi_shape[0],
                    origi_shape[1],
                )
                detection_masks_reframed = tf.cast(
                    detection_masks_reframed > 0.5, tf.uint8).numpy()
                threshold_detections[
                    "detection_masks_reframed"] = detection_masks_reframed[
                        indexes]

            ## Don't show labels if hide labels checkbox is checked
            if f"hide_labels_{each_file_name}" not in st.session_state:
                st.session_state[f"hide_labels_{each_file_name}"] = False
            if not st.session_state[f"hide_labels_{each_file_name}"]:
                output_image = draw_detections(origi_image.copy(), origi_shape,
                                               threshold_detections)
            else:
                output_image = Image.fromarray(origi_image, mode="RGB")

            json_output = get_json_output(threshold_detections)

            ## Create three columns
            col1, col2, col3 = st.columns([0.4, 0.4, 0.2])

            ## Display original image
            with col1:
                st.markdown('<p style="text-align: center;">Original</p>',
                            unsafe_allow_html=True)
                st.image(origi_image, use_column_width="auto")
                st.caption(each_file_name)

            ## Display output image with prediction overlaid
            with col2:
                st.markdown('<p style="text-align: center;">Output</p>',
                            unsafe_allow_html=True)
                st.image(output_image, use_column_width="auto")
                st.checkbox("Hide labels", key=f"hide_labels_{each_file_name}")

                def download_image():
                    buf = BytesIO()
                    output_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    return byte_im

                st.download_button(
                    "Download image",
                    data=download_image(),
                    file_name=f"{each_file_name.split('.')[0]}_output.png",
                    mime="image/png",
                    key=f"download_{each_file_name}")

            ## Display JSON output of predictions
            with col3:
                st.markdown('<p style="text-align: center;">JSON Output</p>',
                            unsafe_allow_html=True)
                st.download_button(
                    label="Download JSON",
                    file_name="predictions_output.json",
                    mime="application/json",
                    data=json.dumps(json_output),
                )
                st.json(json_output, expanded=False)


if __name__ == "__main__":
    st.set_page_config(page_title="Datature Inference Dashboard",
                       layout="wide")

    if "hub_hash" not in st.session_state:
        st.session_state.hub_hash = ""
    if "local_hash" not in st.session_state:
        st.session_state.local_hash = ""
    if "prediction_cache" not in st.session_state:
        st.session_state.prediction_cache = {}

    st.header("Datature Inference Dashboard")
    st.write("Please fill in the project secret and model key to get started.")

    st.radio("Select model loading method", ["Hub", "Local"], key="method")
    if st.session_state.method == "Local":
        st.text_input("Model Path", key="model_path")
        st.text_input("Label Map Path", key="label_map_path")

        if len(st.session_state.model_path) > 0 and len(
                st.session_state.label_map_path) > 0:
            load_local_model()
            upload_images()
            predict_all()
            visualise()

    elif st.session_state.method == "Hub":
        st.text_input("Project Secret", key="project_secret")
        st.text_input("Model Key", key="model_key")

        if (len(st.session_state.model_key) > 0
                and len(st.session_state.project_secret) > 0):
            load_hub_model()
            upload_images()
            predict_all()
            visualise()
