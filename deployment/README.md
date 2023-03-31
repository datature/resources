# Deployment

This section contains scripts for deploying your trained model in various ways. We currently support the following deployment methods:

- [Inference API](inference-api/), for hosting your models on our servers and performing inference through API calls.
- [Edge](edge/), for deploying your models on edge devices such as Raspberry Pi.
- [Local Inference](local-inference/), for running inference on your local machine.

## Getting Started

Please refer to the `README.md` file in each folder for more information on how to use the scripts.
## Download Model

Users can download their trained model directly from [Nexus](https://nexus.datature.io/) or port the trained model through Datature Hub. Users need two sets of keys for the second method: `Model Key` and `Project Secret Key`.<br>

### Model Key

To convert that artifact into an exported model for the prediction service, in Nexus, select `Artifacts` under Project Overview. Within the artifacts page, select your chosen artifact and model format to generate a model key for deployment by clicking the triple dots box shown below.

![modelkey](/assets/modelkey.png)

### Project Secret Key

You can generate the project secret key on the Nexus platform by going to `API Management` and selecting the `Generate New Secret` button, as shown below.

![projectsecret](/assets/projectsecret.png)
