# Datature Edge

[![Join Datature Slack](https://img.shields.io/badge/Join%20The%20Community-Datature%20Slack-blueviolet)](https://datature.io/community)

> **NOTE**:
> [Datature Edge](https://github.com/datature/edge) has its own repository now! Please refer there for the full source code and latest updates.

**Edge pushes your inference pipeline to the very edge**. Most often than not, we do not have access to servers or cloud resources to deploy our models. We created Edge to help engineers and teams to run inference on-premise, so say goodbye to expensive GPUs and high latencies.

Edge is an open-source inference framework primarily written in Python. It is heavily designed to be used with [Nexus, our MLOps platform](https://www.datature.io/nexus), but can be used independently and integrated with your own MLOps pipeline.

Made with `♥` by [Datature](https://datature.io)

<br>
<p align="center">
  <img alt="Edge Inference" src="https://github.com/datature/edge/blob/raspberry-pi/docs/images/raspberry-pi/rbc_raspi.gif?raw=true" width="90%">
</p>

## Table of Contents

- [Datature Edge](#datature-edge)
  - [Table of Contents](#table-of-contents)
  - [Supported Devices](#supported-devices)
  - [Setting up Edge](#setting-up-edge)
  - [Running Edge](#running-edge)
  - [Environment \& Configuration](#environment--configuration)
  - [Customizing Edge](#customizing-edge)
    - [Input Module](#input-module)
    - [Preprocessing Module](#preprocessing-module)
    - [Inference Module](#inference-module)
      - [Model Loading](#model-loading)
      - [Model Predictions](#model-predictions)
    - [Postprocessing Module](#postprocessing-module)
    - [Output Module](#output-module)
  - [Contributing](#contributing)

## Supported Devices

Currently, Edge supports the following devices:
| Device            | OS                      |
| ----------------- | ----------------------- |
| CPU               | Ubuntu 22.04            |
| Raspberry Pi 4b   | 32-bit Raspbian Buster  |

## Setting up Edge

We provide convenient setup scripts for each supported device that will install all the necessary dependencies for you. This will run a universal bash script that installs all the necessary dependencies for Edge to run on your device. The script will also install the Edge system script, which will automatically start Edge on boot.

```bash
git clone https://github.com/datature/edge.git
chmod u+x build.sh
./build.sh
```

## Running Edge

Once you have installed Edge, you can run a demo to check if everything is working fine using the following command:

```bash
datature-edge --start
```

This will load a sample MobileNet TFLite model that detects red blood cells and run inference on a camera feed. If the script has successfully started, you should see a window displaying the camera feed.

Note: you may need to prepend `sudo` to the commands if you are not running as root.

To stop the Edge system script, run the following command:

```bash
datature-edge --stop
```

If you want to disable automatic restarts, run the following command:

```bash
datature-edge --disable
```

To re-enable automatic restarts, run the following command:

```bash
datature-edge --enable
```

To view all other available options, run the following command:

```bash
datature-edge --help
```

## Environment & Configuration

You can choose how you want to set up your inference pipeline based on your needs. Each pipeline consists of <b>one</b> input module, optional preprocessing modules, <b>one</b> inference module, optional postprocessing modules and optional output modules. If you do not require any custom modules, you can specify [module configurations](https://github.com/datature/edge/src/edge/python/common/samples/config/config1.yaml) through a YAML file. Otherwise, you will need to wrap your custom functions inside an inherited class from the [abstract modules](#customizing-edge) provided.

Datature Edge runs in an isolated environment with configuration variables. Hence, you can run multiple instances of Edge on the same device without any conflicts, as long as your device has enough resources to support the multiple processes. To change the environment variables or add/remove instances, you will need to create a folder containing environment configuration files. Each file can be of any file extension for text (e.g. `.txt`, `.conf`, etc.) and should contain the following variables:

```bash
DATATURE_EDGE_ROOT_DIR=<DIR>/edge/
DATATURE_EDGE_PYTHON_CONFIG=<CONFIG_FILE_PATH>
```

where `<DIR>` is the parent directory where the Edge repository is located, and `<CONFIG_FILE_PATH>` is the path to the YAML configuration file.

To restart Edge with the new environment variables, run the following command:

```bash
datature-edge --config <ENV_CONFIG_DIR>
```

## Customizing Edge

 The following sections will guide you through the process of customizing your pipeline.

| Module            | Abstract Class Template File                                                                                                                                      |
| :--- | :--- |
| Input             | [`src/edge/python/core/devices/abstract_input.py`](https://github.com/datature/edge/src/edge/python/core/devices/abstract_input.py)                                                                |
| Preprocessing     | [`src/edge/python/core/components/data/preprocessors/abstract_preprocessor.py`](https://github.com/datature/edge/src/edge/python/core/components/data/preprocessors/abstract_preprocessor.py) |
| Postprocessing    | [`src/edge/python/core/components/data/postprocessors/abstract_postprocessor.py`](https://github.com/datature/edge/src/edge/python/core/components/data/postprocessors/abstract_postprocessor.py)  |
| Output            | [`src/edge/python/core/devices/abstract_output.py`](https://github.com/datature/edge/src/edge/python/core/devices/abstract_output.py)                                                              |

### Input Module

By default, Edge supports the following input modules:

| Input Module      | Description                       |
| ----------------- | --------------------------------- |
| Webcam            | Camera streaming                  |
| Image             | Single image (png, jpg, jpeg)     |
| Video             | Video file                        |

To include your own custom input module, you can create a class that inherits from the [`AbstractInput`](./src/edge/python/core/devices/abstract_input.py) class and override the `run` and `stop` methods. The module should also contain two attributes, `frame` and `frame_id` to return a numpy array of shape `NHWC`, and a string representing a unique frame id, respectively.

```python
class CustomInput(AbstractInput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self):
        # Run input module and update self.frame and self.frame_id
        # You can start a thread to continuously update self.frame and self.frame_id
        # (e.g. for camera streaming)

        # Numpy array of shape NHWC
        self.frame = frame

        # Unique frame id
        self.frame_id = frame_id

    def stop(self):
        # Stop input module and clean up resources
        pass
```

### Preprocessing Module

Preprocessing modules are used to perform any preprocessing steps on the input data before it is passed to the inference module. This step is optional if you choose to perform preprocessing in the input module. By default, Edge supports the following preprocessing modules:

| Preprocessing Module | Description                                |
| -------------------- | ------------------------------------------ |
| Resize               | Resize image to specified width and height |

To include your own custom preprocessing module, you can create a class that inherits from the [`AbstractPreprocessor`](./src/edge/python/core/components/data/preprocessors/abstract_preprocessor.py) class and override the `run` method. This method accepts a dictionary of assets containing the image data to be processed under the key `input_frame`. The returned data should be a dictionary containing the preprocessed image data under the key `input_frame`.

```python
class CustomPreprocessor(AbstractPreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, assets):
        # Preprocess image
        assets["input_frame"] = preprocess(assets["input_frame"])
        return assets
```

### Inference Module

Edge supports all models trained on Datature [Nexus](https://www.datature.io/nexus). You can load your model from Nexus using Datature Hub by providing your [project secret](https://developers.datature.io/docs/hub-and-api#locating-the-project-secret) and your corresponding [model key](https://developers.datature.io/docs/hub-and-api#locating-the-model-key). Alternatively, you can load your own custom model locally by providing the path to your model file and the path to your label file. The model formats that we currently support are listed below:

<u>Object Detection</u>

| Model Architecture  | Description               |
| ------------------- | ------------------------- |
| MobileNet           | Tensorflow, TFLite, ONNX  |
| EfficientDet        | Tensorflow, TFLite, ONNX  |
| ResNet              | Tensorflow, TFLite, ONNX  |
| FasterRCNN          | Tensorflow, TFLite, ONNX  |
| YOLOv4              | Tensorflow, ONNX          |
| YOLOX               | ONNX                      |

<u>Segmentation</u>

| Model Architecture  | Description |
| ------------------- | ----------- |
| MaskRCNN            | ONNX        |
| DeepLabV3           | ONNX        |
| FCN                 | ONNX        |
| UNet                | ONNX        |

If you wish to build your own inference module, you can create a class that inherits from the [`InferenceEngine`](./src/edge/python/core/components/inference/engine.py) class. This class should contain a [Loader](#model-loading) class and a [Predictor](#model-predictions) class to load the model and run the prediction function respectively.

Note: Creating a custom inference module is for advanced users only, since you will likely need to customize other modules in the inference pipeline to support your custom model. If you are unsure about how to proceed, feel free to join our [Community Slack](https://datature.io/community?ref=datature-blog) to post questions and get help from our team.

```python
class CustomInferenceEngine(InferenceEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialize custom loader
        self.loader = CustomLoader(**kwargs)

        # Initialize custom predictor
        self.predictor = CustomPredictor(**kwargs)
```

#### Model Loading

If you wish to load any custom models that are not supported by Edge, you can create a class that inherits from the [`AbstractLoader`](./src/edge/python/core/components/inference/loaders/abstract_loader.py) class and override the `load_model` method. This method sets the `model` attribute to the loaded model. You can also set the `category_index` attribute to a dictionary of class labels of your model if you wish to utilize the class label mapping for any post-processing steps.

```python
class CustomLoader(AbstractLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self):
        # Load model
        self.model = load_model(self.model_path)

        # Load class labels
        self.category_index = load_labels(self.label_path)
```

#### Model Predictions

To allow your custom model to generate predictions, you can create a class that inherits from the [`AbstractPredictor`](./src/edge/python/core/components/inference/predictors/abstract_predictor.py) class and override the `predict` method. This method accepts an image frame to run inference on. The returned data should be a dictionary containing the inference results.

```python
class CustomPredictor(AbstractPredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, frame):
        # Run inference on frame
        # Change to your model's prediction function
        predictions_output = self.model(frame)
        return predictions_output
```

### Postprocessing Module

Postprocessing modules are used to perform any postprocessing steps on the inference results before it is passed to the output module. This step is optional if you choose to perform postprocessing in the inference or output modules.

To include your own custom postprocessing module, you can create a class that inherits from the [`AbstractPostprocessor`](./src/edge/python/core/components/data/postprocessors/abstract_postprocessor.py) class and override the `run` method. This method accepts a dictionary of assets containing the inference results to be processed under the key `predictions`. The returned data should be a dictionary containing the postprocessed inference results under the key `predictions`.

```python
class CustomPostprocessor(AbstractPostprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, assets):
        # Postprocess predictions
        assets["predictions"] = postprocess(assets["predictions"])
        return assets
```

### Output Module

Output modules are used for interfacing with other applications or peripherals, such as visualization, sending data through API calls, and others. By default, Edge supports the following output modules:

| Output Module       | Description                                             |
| ------------------- | ------------------------------------------------------- |
| Save                | Save inference results to disk                          |
| Visualization       | Visualize inference results                             |
| Datature Python SDK | Upload images and inference results for active learning |

## Contributing

We welcome contributions from the community to expand support for more devices and modules. Please refer to our [Contributing Guide](CONTRIBUTING.md) for more information.
