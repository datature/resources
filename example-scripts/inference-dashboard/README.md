# Building an Inference Dashboard with Datature Hub & Streamlit

This repository contains the full code for building a simple inference dashboard with Streamlit, Tensorflow, and OpenCV. The tutorial can be found [here](https://www.datature.io/blog/building-a-simple-inference-dashboard-with-streamlit).

## Getting Started

### Environment Setup

Python 3.7 <= version <= Python 3.9<br>
Tensorflow >= 2.5.0

### Installing Packages

To install necessary packages like Datature Hub, Streamlit, Tensorflow, and OpenCV, run the command below. It is recommended that you do this in a virtual environment of your choice (such as <i>virtualenv</i> or <i>virtualenvwrapper</i>).

```bash
pip install -r requirements.txt
```

### Running the App

To start the app, run the command below.

```bash
streamlit run inference_dashboard.py
```

The dashboard should be running on your local machine at http://localhost:8501. The dashboard currently supports inference with Tensorflow Object Detection models exported from Datature Nexus. There are two ways to load a model into the dashboard.

#### Using Datature SDK

![sdk](/example-scripts/inference-dashboard/assets/load_sdk_model.png)

We leverage Datature SDK to export and download your model from Nexus. To do so, ensure that you have an exported Tensorflow artifact in your Nexus project. You can then enter your [secret key](https://developers.datature.io/docs/hub-and-api) in the `Project Secret Key` field provided. A list of all Tensorflow exported artifacts will be listed in the `Select Model` dropdown. Select the model you wish to load and it will automatically be downloaded and loaded into the dashboard.

#### Loading from Local Directory

![local](/example-scripts/inference-dashboard/assets/load_local_model.png)

Alternatively, you can load a model from a local directory. To do so, enter the path to your Tensorflow SavedModel directory (`<DIR>/saved_model`) in the `Model Path` field provided. Please also enter the path to the label map file (`<DIR>/label_map.pbtxt`) in the `Label Map Path` field.

For both methods, you will also need to include the model input size as a comma-separated string of `WIDTH,HEIGHT` in the `Model Input Size` field. For example, if your model input size is 640x640, you should enter `640,640` in the field. Once the model has been loaded successfully, you can then upload image(s) to run inference on and visualise or download the results.

![loaded](/example-scripts/inference-dashboard/assets/loaded_model.png)

![predictions](/example-scripts/inference-dashboard/assets/predictions.png)

To stop the app, press `Ctrl+C` in the terminal.
