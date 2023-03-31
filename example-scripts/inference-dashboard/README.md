# Building an Inference Dashboard with Datature Hub & Streamlit

This repository contains the full code for building a simple inference dashboard with Datature Hub, Streamlit, Tensorflow, and OpenCV. The tutorial can be found [here](https://www.datature.io/blog/building-a-simple-inference-dashboard-with-streamlit).

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

The dashboard should be running on your local machine at http://localhost:8501. You can then provide the project secret and model key in the fields to load your desired Tensorflow model. With the model loaded, you can then upload image(s) to run inference on and visualise or download the results.

To stop the app, press `Ctrl+C` in the terminal.
