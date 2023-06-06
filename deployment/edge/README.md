# Datature Edge

[![Join Datature Slack](https://img.shields.io/badge/Join%20The%20Community-Datature%20Slack-blueviolet)](https://datature.io/community)

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
    - [Automatic Installation](#automatic-installation)
    - [Manual Installation](#manual-installation)
    - [Cleaning Up](#cleaning-up)
  - [Running Edge](#running-edge)
    - [Python Script](#python-script)
    - [Daemon Service](#daemon-service)
    - [Environment \& Configuration](#environment--configuration)
    - [Troubleshooting](#troubleshooting)
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

| Device          | OS                                  |
| --------------- | ----------------------------------- |
| CPU             | Ubuntu 20.04 (Focal), 22.04 (Jammy) |
| Raspberry Pi 4b | 32-bit Raspbian Buster              |
| Jetson Orin     | NVIDIA Jetson Linux 35.3.1          |

If you wish to run Edge on a different device, please refer to the [manual installation](#manual-installation) section.

## Setting up Edge

### Automatic Installation

<details>
     <summary>Click to expand</summary><br>

We provide convenient setup scripts for each supported device that will install all the necessary dependencies for you. This will run a universal bash script that installs all the necessary dependencies for Edge to run on your device. Please ensure that you have sudo privileges before running the script, and that your current directory is the root of the repository (e.g. `/home/$USER/edge/`).

```bash
git clone https://github.com/datature/edge.git
chmod u+x build.sh
source build.sh
```

The script will also install some system files, which will automatically start Edge on boot.

- `/usr/bin/datature-edge`: The main binary executable for Datature Edge.
- `/etc/datature-edge.conf`: The root configuration file for Datature Edge.
- `/etc/systemd/system/datature-edge.service`: The systemd service file for Datature Edge.

</details>

### Manual Installation

If you wish to install Datature Edge on an unsupported device, you can follow a rough guide at `src/edge/python/setup/cpu/jammy/setup.sh` and see the necessary dependencies that you need to install on your own device. Do note that Datature Edge is only tested on the supported devices, so we cannot guarantee that it will work on your device. Alternatively, you can create an issue on GitHub for us to add to our development roadmap.

### Cleaning Up

If you wish to remove the system files, you can run the following commands:

```bash
chmod u+x clean.sh
source clean.sh
```

## Running Edge

### Python Script

<details>
     <summary>Click to expand</summary><br>

Datature Edge can be run by creating a simple Python script. The following is a sample script that loads a MobileNet TFLite model that detects red blood cells and runs inference on a camera feed on a laptop CPU running Ubuntu 22.04. You can find both the script and the configuration file in the `samples/python` folder.

`main.py`

```python
import os

from common.logger import Logger
from core.devices import DeviceEngine

## Set Configuration File Path
os.environ["DATATURE_EDGE_PYTHON_CONFIG"] = "./samples/python/config.yaml"

try:
    device_engine = DeviceEngine(config=True)
    device_engine.run()
except Exception as exc:
    Logger.error(f"{exc.__class__.__name__}: {exc}")
```

`config.yaml`

```yaml
name: my_inference_config
device: cpu

inference:
  detection_type: object_detection
  model_format: tflite
  model_architecture: mobilenet

  model_path: ./src/edge/python/common/samples/models/tflite/model.tflite
  label_path: ./src/edge/python/common/samples/label.txt

  input_shape: [320, 320]
  threshold: 0.7

  bbox_format: [ymin, xmin, ymax, xmax]

blocks:
  input:
    name: webcam
    device: -1
    frame_size: [640, 480]
    fps: 32
    max_buffer: 100

  preprocessors:
    modules:
      - transforms:
          tools:
            - resize:
                shape: [320, 320]

  postprocessors:
    modules: []

  output:
    modules:
      - opencv:
          type: video_show
          window_name: Datature Edge
          frame_size: [640, 480]

debug:
  active: true
  log_folder: ./logs/debug/

profiling:
  active: false
  log_folder: ./logs/profiling/
```

For more information on how to modify the YAML configuration file, please refer to the [configuration documentation](docs/configuration.md).

</details>

### Daemon Service

<details>
     <summary>Click to expand</summary><br>

Datature Edge can run as a daemon service in the background and automatically restarts if it crashes. This is useful if you want to run Edge on boot and not have to worry about manually restarting it if it crashes. Do note that the daemon service is currently only available for supported devices that run a Linux OS.

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

### Environment & Configuration

You can choose how you want to set up your inference pipeline based on your needs. Each pipeline consists of <b>one</b> input module, optional preprocessing modules, <b>one</b> inference module, optional postprocessing modules and optional output modules. If you do not require any custom modules, you can specify [module configurations](./src/edge/python/common/samples/config/config1.yaml) through a YAML file. Otherwise, you will need to wrap your custom functions inside an inherited class from the [abstract modules](#customizing-edge) provided.

Datature Edge runs in an isolated environment with configuration variables. Hence, you can run multiple instances of Edge on the same device without any conflicts, as long as your device has enough resources to support the multiple processes. To change the environment variables or add/remove instances, you will need to create a folder containing environment configuration files.

```bash
|-- <ENV_CONFIG_DIR>
   |-- <ENV_CONFIG_FILE_1>
   |-- <ENV_CONFIG_FILE_2>
   |-- ...
```

Each file can be of any file extension for text (e.g. `.txt`, `.conf`, etc.) and should contain the following variables:

`ENV_CONFIG_FILE_1`:

```bash
# Full path to the installation path of Datature Edge repository
DATATURE_EDGE_ROOT_DIR=<PARENT_DIR>/edge/
# Full path to a YAML configuration file
DATATURE_EDGE_PYTHON_CONFIG=<CONFIG_FILE_PATH>
```

For more information on how to modify the YAML configuration file, please refer to the [configuration documentation](docs/configuration.md).

To restart Edge with the new environment variables, run the following command:

```bash
datature-edge --config <ENV_CONFIG_DIR>
```

</details>

### Troubleshooting

If you encounter any issues while running Edge, you can check the debug logs in your specified log folder (this defaults to `<EDGE_DIR>/logs/debug/debug.log`). To enable logging, you will need to set the `debug.active` variable to `true` in your configuration file.

If the debug logs do not offer any informative details, it could be a system-level issue. You can run the following commands in your terminal:

- `datature-edge --status`: check if the Edge system script is running
- `journalctl -u datature-edge`: check the system logs for the Edge system script

If you are still unable to resolve the issue, you can open a [GitHub issue](https://github.com/datature/edge/issues) and we will get back to you as soon as possible.

## Customizing Edge

The following sections will guide you through the process of customizing your pipeline with custom modules.

| Module            | Abstract Class Template File                                                                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Input             | [`src/edge/python/core/devices/abstract_input.py`](./src/edge/python/core/devices/abstract_input.py)                                                                |
| Preprocessing     | [`src/edge/python/core/components/data/preprocessors/abstract_preprocessor.py`](./src/edge/python/core/components/data/preprocessors/abstract_preprocessor.py)      |
| Postprocessing    | [`src/edge/python/core/components/data/postprocessors/abstract_postprocessor.py`](./src/edge/python/core/components/data/postprocessors/abstract_postprocessor.py)  |
| Output            | [`src/edge/python/core/devices/abstract_output.py`](./src/edge/python/core/devices/abstract_output.py)                                                              |

### Input Module

<details>
     <summary>Click to expand</summary><br>

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
        ...

    def run(self):
        # Run input module and update self.frame and self.frame_id
        # You can start a thread to continuously update self.frame and self.frame_id
        # (e.g. for camera streaming)
        ...

        # Frame should be a numpy array of shape NHWC
        self.frame = frame

        # Unique frame id, e.g. UNIX timestamp or UUID string
        self.frame_id = frame_id

    def load_data(self, assets: dict):
        # Load data into a dictionary of assets
        # This method is called by the engine to retrieve data from the input module
        # The assets dictionary must contain the following keys for the inference module:
        assets["input_frame"] = self.frame
        assets["frame_id"] = self.frame_id

        # You can add optional keys to the assets dictionary that can be used in
        # preprocessing, postprocessing, and output modules
        ...

    def stop(self):
        # Stop input module and clean up resources if necessary
        self.stopped = True
        ...
```

</details>

### Preprocessing Module

<details>
     <summary>Click to expand</summary><br>

Preprocessing modules are used to perform any preprocessing steps on the input data before it is passed to the inference module. This step is optional if you choose to perform preprocessing in the input module. By default, Edge supports the following preprocessing modules:

| Preprocessing Module | Description                                |
| -------------------- | ------------------------------------------ |
| Resize               | Resize image to specified width and height |

To include your own custom preprocessing module, you can create a class that inherits from the [`AbstractPreprocessor`](./src/edge/python/core/components/data/preprocessors/abstract_preprocessor.py) class and override the `run` method. This method accepts a dictionary of assets containing the image data to be processed under the key `input_frame`. The returned data should be a dictionary containing the preprocessed image data under the key `input_frame`. The keys are specified such that multiple preprocessing modules can be chained together, and there is a common output key for the inference module to access the preprocessed input data.

```python
class CustomPreprocessor(AbstractPreprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ...

    def run(self, assets: dict):
        # Preprocess image
        assets["input_frame"] = preprocess(assets["input_frame"])
        return assets
```

</details>

### Inference Module

<details>
     <summary>Click to expand</summary><br>

Edge supports all models trained on Datature [Nexus](https://www.datature.io/nexus). You can load your model from Nexus using Datature Hub by providing your [project secret](https://developers.datature.io/docs/hub-and-api#locating-the-project-secret) and your corresponding [model key](https://developers.datature.io/docs/hub-and-api#locating-the-model-key). Alternatively, you can load your own custom model locally by providing the path to your model file and the path to your label file. The model formats that we currently support are listed below:

<u>Object Detection</u>

| Model Architecture  | Model Format(s)                   |
| ------------------- | --------------------------------- |
| MobileNet           | Tensorflow, TFLite, ONNX          |
| EfficientDet        | Tensorflow, TFLite, ONNX          |
| ResNet              | Tensorflow, TFLite, ONNX          |
| FasterRCNN          | Tensorflow, TFLite, ONNX          |
| YOLOv4              | Tensorflow, TFLite, ONNX, PyTorch |
| YOLOX               | Tensorflow, TFLite, ONNX, PyTorch |

<u>Instance Segmentation</u>

| Model Architecture  | Model Format(s)           |
| ------------------- | ------------------------- |
| MaskRCNN            | Tensorflow, TFLite, ONNX  |

<u>Semantic Segmentation</u>

| Model Architecture  | Model Format(s)                   |
| ------------------- | --------------------------------- |
| DeepLabV3           | Tensorflow, TFLite, ONNX, PyTorch |
| FCN                 | Tensorflow, TFLite, ONNX, PyTorch |
| UNet                | Tensorflow, TFLite, ONNX, PyTorch |

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
        ...

    def load(self):
        # Load model
        # `self.model_path` is a string containing the path to your model file,
        # defined in your configuration YAML file
        self.model = load_model(self.model_path)

        # Load class labels
        # `self.label_path` is a string containing the path to your label map file,
        # defined in your configuration YAML file
        self.category_index = load_labels(self.label_path)
```

#### Model Predictions

To allow your custom model to generate predictions, you can create a class that inherits from the [`AbstractPredictor`](./src/edge/python/core/components/inference/predictors/abstract_predictor.py) class and override the `predict` method. This method accepts an image frame in a model input-compatible format (such as a Tensor or Numpy array) to run inference on. Please refer to the individual model framework documentation on what input format your custom model accepts for inference. The returned data should be a dictionary containing the inference results.

```python
class CustomPredictor(AbstractPredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ...

    def predict(self, frame):
        # Run inference on frame
        # Change to your model's prediction function
        predictions_output = self.model(frame)
        return predictions_output
```

</details>

### Postprocessing Module

<details>
     <summary>Click to expand</summary><br>

Postprocessing modules are used to perform any postprocessing steps on the inference results before it is passed to the output module. This step is optional if you choose to perform postprocessing in the inference or output modules.

To include your own custom postprocessing module, you can create a class that inherits from the [`AbstractPostprocessor`](./src/edge/python/core/components/data/postprocessors/abstract_postprocessor.py) class and override the `run` method. This method accepts a dictionary of assets containing the inference results to be processed under the key `predictions`. The returned data should be a dictionary containing the postprocessed inference results under the key `predictions`. The keys are specified such that multiple postprocessing modules can be chained together, and there is a common output key for the output module to access the inference results.

```python
class CustomPostprocessor(AbstractPostprocessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ...

    def run(self, assets: dict):
        # Postprocess predictions
        assets["predictions"] = postprocess(assets["predictions"])
        return assets
```

</details>

### Output Module

<details>
     <summary>Click to expand</summary><br>

Output modules are used for interfacing with other applications or peripherals, such as visualization, sending data through API calls, and others. By default, Edge supports the following output modules:

| Output Module       | Description                                             |
| ------------------- | ------------------------------------------------------- |
| Save                | Save inference results to disk                          |
| Visualization       | Visualize inference results                             |
| Datature Python SDK | Upload images and inference results for active learning |

To include your own custom output module, you can create a class that inherits from the [`AbstractOutput`](./src/edge/python/core/devices/abstract_output.py) class and override the `run` method. This method accepts accepts a dictionary of assets that contains the preprocessed input data pre-inference and the postprocessed inference results. This allows you to utilize all the data available in the inference pipeline to perform your output operations.

```python
class CustomOutput(AbstractOutput):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        ...

    def run(self, assets: dict):
        # Perform output operations
        ...
```

</details>

## Contributing

We welcome contributions from the community to expand support for more devices and modules. Please refer to our [Contributing Guide](CONTRIBUTING.md) for more information.
