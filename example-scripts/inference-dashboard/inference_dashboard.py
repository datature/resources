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
@Desc    :   Datature inference dashboard built using Streamlit for prediction visualisations.
'''

import json
import os
from io import BytesIO
from zipfile import ZipFile

import absl.logging
import cv2
import datature
import numpy as np
import streamlit as st
import tensorflow as tf
import wget
from helper import load_image_into_numpy_array, load_label_map, nms_boxes
from PIL import Image
from streamlit import session_state as state

## Disable unnecessary warnings for tensorflow
absl.logging.set_verbosity(absl.logging.ERROR)
## Comment out next line to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
## Comment out to set verbose to true
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_local_model():
    """Load model locally using model path and label map path."""
    if state.local_hash != state.model_path + state.label_map_path:
        with st.spinner('Loading model...'):
            ## Load Tensorflow model
            state.trained_model = tf.saved_model.load(
                state.model_path).signatures["serving_default"]
            state.output_name = list(
                state.trained_model.structured_outputs.keys())[0]

            ## Load label map
            state.category_index = load_label_map(state.label_map_path)

            ## Load color map
            state.color_map = {}
            for each_class in range(len(state["category_index"])):
                state.color_map[each_class] = [
                    int(i) for i in np.random.choice(range(256), size=3)
                ]
        state.local_hash = state.model_path + state.label_map_path
        st.success('Model loaded successfully!')


def load_sdk_model():
    """Load model from Nexus using Datature SDK with project secret."""
    if "artifact_id" not in state or state.artifact != state.artifact_names[
            state.artifact_id]:
        state.artifact_id = [
            key for key in state.artifact_names
            if state.artifact_names[key] == state.artifact
        ][-1]
        exported_artifact = list(
            datature.Artifact.list_exported(state.artifact_id))[-1]

        url = exported_artifact["download"]["url"]
        dir_name = os.path.join(os.path.expanduser("~"), ".datature",
                                url.split("/")[-1].split(".")[0])
        out_fname = f"{dir_name}.zip"

        if state.local_hash != state.secret_key + dir_name:
            with st.spinner("Loading model..."):
                if not os.path.isdir(dir_name):
                    wget.download(url, out=out_fname)
                    os.mkdir(dir_name)
                    with ZipFile(out_fname, "r") as zip_ref:
                        zip_ref.extractall(dir_name)

                ## Load Tensorflow model
                state.trained_model = tf.saved_model.load(
                    os.path.join(dir_name,
                                 "saved_model")).signatures["serving_default"]
                state.output_name = list(
                    state.trained_model.structured_outputs.keys())[0]

                ## Load label map
                state.category_index = load_label_map(
                    os.path.join(dir_name, "label_map.pbtxt"))

                ## Load color map
                state.color_map = {}
                for each_class in range(len(state["category_index"])):
                    state.color_map[each_class] = [
                        int(i) for i in np.random.choice(range(256), size=3)
                    ]
        state.local_hash = state.secret_key + dir_name
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

    state.file_names = [each_file.name for each_file in state.uploaded_imgs]


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
    width, height = state.model_input_size.split(",")
    image_resized, origi_shape = load_image_into_numpy_array(
        image, int(height), int(width))
    input_image = np.expand_dims(image_resized.astype(np.float32), 0)

    ## Feed image into model
    detections_output = state.trained_model(inputs=input_image)
    detections_output = np.array(detections_output[state.output_name][0])

    return np.array(image), origi_shape, detections_output


def predict_all():
    """Predict all the uploaded images and store the predictions in the session state."""
    with st.spinner('Running predictions...'):
        for each_file in state.uploaded_imgs:
            if each_file.name not in state.prediction_cache.keys():
                origi_image, origi_shape, detections = predict(each_file)

                state.prediction_cache.update({
                    each_file.name: {
                        "img": origi_image,
                        "shape": origi_shape,
                        "detections": detections
                    }
                })


def draw_detections(origi_image, origi_shape, detections):
    """Draw the predictions on the image.
    Args:
        origi_image: original image
        origi_shape: shape of the original image (img_height, img_width)
        detections: model predictions
    Returns:
        Output image with predictions drawn on it.
    """
    for each_bbox, each_class, each_score in list(
            zip(detections["boxes"], detections["classes"],
                detections["scores"])):
        color = state.color_map.get(each_class - 1)

        ## Draw bounding box
        cv2.rectangle(
            origi_image,
            (
                int(each_bbox[1] * origi_shape[0]),
                int(each_bbox[0] * origi_shape[1]),
            ),
            (
                int(each_bbox[3] * origi_shape[0]),
                int(each_bbox[2] * origi_shape[1]),
            ),
            color,
            2,
        )
        ## Draw label background
        cv2.rectangle(
            origi_image,
            (
                int(each_bbox[1] * origi_shape[0]),
                int(each_bbox[2] * origi_shape[1]),
            ),
            (
                int(each_bbox[3] * origi_shape[0]),
                int(each_bbox[2] * origi_shape[1] + 15),
            ),
            color,
            -1,
        )
        ## Insert label class & score
        cv2.putText(
            origi_image,
            "Class: {}, Score: {}".format(
                str(state.category_index[each_class]["name"]),
                str(round(each_score, 2)),
            ),
            (
                int(each_bbox[1] * origi_shape[0]),
                int(each_bbox[2] * origi_shape[1] + 10),
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
            zip(detections["boxes"], detections["classes"],
                detections["scores"])):
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
                "name": str(state.category_index[each_class]["name"])
            }
        }

        json_output["predictions"].append(prediction)

    return json_output


def visualise():
    """Perform visualisation by filtering out predictions with scores below a threshold,
    drawing the thresholded predictions on the image and generating JSON output."""
    for each_file_name in state.prediction_cache.keys():
        if each_file_name in state.file_names:
            st.slider(f"Threshold_{each_file_name}",
                      min_value=0.0,
                      max_value=1.0,
                      value=0.7,
                      step=0.01,
                      label_visibility="collapsed",
                      help="Threshold",
                      key=f"threshold_{each_file_name}")

            origi_image = state.prediction_cache[each_file_name]["img"]
            origi_shape = state.prediction_cache[each_file_name]["shape"]
            detections = state.prediction_cache[each_file_name]["detections"]

            ## Filter detections
            slicer = detections[:, -1]
            output = detections[:, :6][slicer != 0]
            scores = output[:, 4]
            output = output[scores > state[f"threshold_{each_file_name}"]]
            classes = output[:, 5]
            output = output[classes != 0]

            ## Postprocess detections
            boxes = output[:, :4]
            classes = output[:, 5].astype(np.int32)
            scores = output[:, 4]
            boxes[:, 0], boxes[:, 1] = (boxes[:, 1] * origi_shape[1],
                                        boxes[:, 0] * origi_shape[0])
            boxes[:, 2], boxes[:, 3] = (boxes[:, 3] * origi_shape[1],
                                        boxes[:, 2] * origi_shape[0])
            boxes, classes, scores = nms_boxes(boxes, classes, scores, 0.1)
            boxes = [[
                bbox[1] / origi_shape[1],
                bbox[0] / origi_shape[0],
                bbox[3] / origi_shape[1],
                bbox[2] / origi_shape[0],
            ] for bbox in boxes]  # y1, x1, y2, x2

            threshold_detections = {
                "boxes": boxes,
                "classes": classes,
                "scores": scores
            }

            ## Don't show labels if hide labels checkbox is checked
            if f"hide_labels_{each_file_name}" not in state:
                state[f"hide_labels_{each_file_name}"] = False
            if not state[f"hide_labels_{each_file_name}"]:
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

    if "local_hash" not in state:
        state.local_hash = ""
    if "prediction_cache" not in state:
        state.prediction_cache = {}

    st.header("Datature Inference Dashboard")
    st.write("Please fill in the project secret and model key to get started.")

    st.radio("Select model loading method", ["SDK", "Local"], key="method")
    st.text_input("Model Input Size (WIDTH,HEIGHT)", key="model_input_size")

    if state.method == "Local":
        st.text_input("Model Path", key="model_path")
        st.text_input("Label Map Path", key="label_map_path")

        if len(state.model_path) > 0 and len(state.label_map_path) > 0 and len(
                state.model_input_size) > 0:
            load_local_model()
            upload_images()
            predict_all()
            visualise()

    elif state.method == "SDK":
        st.text_input("Project Secret Key", key="secret_key")

        if len(state.secret_key) > 0:
            if "artifact_names" not in state:
                datature.secret_key = state.secret_key
                artifacts = datature.Artifact.list()
                state.artifact_names = {
                    each_artifact["id"]:
                    f"{each_artifact['flow_title']}: {each_artifact['artifact']}"
                    for each_artifact in artifacts
                    if "tensorflow" in each_artifact["exportable_formats"]
                }
            st.selectbox("Select model",
                         state.artifact_names.values(),
                         key="artifact")

            if state.artifact is not None and len(state.model_input_size) > 0:
                load_sdk_model()
                upload_images()
                predict_all()
                visualise()
