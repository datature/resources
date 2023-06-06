[![Join Datature Slack](https://img.shields.io/badge/Join%20The%20Community-Datature%20Slack-blueviolet)](https://datature.io/community)

<div id="top"></div>

# Datature Script Library

A repository of resources used in our tutorials and guides ⚡️

This library is a collection of useful scripts that can be used for integrating with our platform tools, or for general CV application purposes. The scripts are written in various programming languages and are available under the MIT License.

## Table of Contents

- [Datature Script Library](#datature-script-library)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Usage](#usage)
    - [Contributing](#contributing)
  - [Script Categories](#script-categories)
    - [Example Scripts](#example-scripts)
    - [SDK Guides](#sdk-guides)
    - [Deployment](#deployment)

## Getting Started

### Prerequisites

Firstly, users should clone this repository and change to the resource folder directory.

```bash
git clone https://github.com/datature/resources.git
cd resources
```

In each folder, there will be a `requirements.txt` file that contains the dependencies required for Python scripts to run. Users can install the dependencies by running the following command:

```bash
pip install -r requirements.txt
```

It is recommended to use a virtual environment to install the dependencies. For more information on virtual environments, please refer to [Python venv](https://docs.python.org/3/tutorial/venv.html).

### Usage

Each folder contains a `README.md` file that contains the instructions for running the scripts. Please refer to the `README.md` file for more information.

### Contributing

We welcome contributions to this repository. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for more information on what areas you can contribute in and coding best practice guidelines.

## Script Categories

### Example Scripts

This section contains example scripts that can be used for integrating with our platform tools, or for general CV application purposes.

- [Active Learning](example-scripts/active-learning/), for performing active learning on your dataset.
- [Data Preprocessing](example-scripts/data-preprocessing/), useful tools for preprocessing your data.
- [Inference Dashboard](example-scripts/inference-dashboard/), for easy visualizations of inference results.
- [Learning](example-scripts/learning/), sample scripts for one-shot and few-shot learning.
- [Tracking](example-scripts/tracking/), for single and multi-object tracking in videos.
### SDK Guides

This section contains guides and code snippets on how to use our Datature Python SDK for automating tasks without having to interact with our Nexus platform. The SDK is available on [PyPI](https://pypi.org/project/datature/). It can be installed by running the following command:

```bash
pip install -U datature
```

The SDK can either be invoked in [Python](sdk-guides/python/), or through the [command line interface (CLI)](sdk-guides/cli/). For more information or advanced features on the SDK, please refer to the [SDK documentation](https://developers.datature.io/reference/getting-started).

### Deployment

This section contains scripts on how to deploy your models trained on Nexus for inference. We currently support the following deployment methods:

- [Edge Deployment](deployment/edge/), for deploying models on edge devices such as Raspberry Pi.
- [Inference API](deployment/inference-api/), where models are hosted on our servers and inference can be performed through API calls.
- [Local Inference](deployment/local-inference/), for running simple inference scripts on your local machine.
