# CLI Command Guide

Datature constantly strives to make MLOps accessible for all users, from individual developers to large enterprises with established codebases. With that in mind, one of the tools we are making accessible is our Command Line Interface (CLI), which allows you to easily perform general interactions at all the essential steps of the MLOps pipeline just through simple shell commands.

The main function categories are as follows:

| Function Category                                  | Description                                                                                          |
| :------------------------------------------------- | :--------------------------------------------------------------------------------------------------- |
| [Project Management](doc:project-sdk-functions)    | This gives you the essential functions to getting and changing basic information about your project. |
| [Annotation Management](doc:tag-sdk-functions)     | This deals with the upload and retrieval of annotations stored on the Nexus platform.                |
| [Asset Management](doc:asset-sdk-functions)        | This deals with the upload, retrieval, download, and removal of assets on the Nexus platform.        |
| [Operation Functions](doc:operation-sdk-functions) | This allows users to get insight on processes called in the SDK that are ongoing.                    |

# How to Get Started

## Install Python 3.7 or Above

As this is an SDK for Python, users will need to ensure that they have Python installed. As of now, we currently fully support all versions of Python from 3.7 or above. If you are having issues with the SDK, please ensure that your environment uses an Python version that is fully supported, or else we are not able to guarantee functionality or fixes.

## Install Datature's CLI

To make installation as simple as possible, we have made the Python package available on PIP, Python's most popular package installation tool. After ensuring you have pip installed in your environment, which should come with a standard Python installation, you can simply enter the following command below.

```bash
pip install --upgrade datature
```

## Authentication

The final step that is essential to all successful requests is to ensure that you log on to the platform, access the relevant project, and store the project secret key which can be found in Integrations. As mentioned in the following link, only the project owner or those with relevant permissions can have access to the project secret on the platform. For more detail on the project key and secret key, check out [this link](doc:hub-and-api) for more information.

Once you have the project secret, you will now be able to make API requests using the CLI by entering the command `datature project auth`:

```bash
datature project auth
[?] Enter the project secret: ************************************************
[?] Make [Your Project Name] the default project? (Y/n): y
Authentication succeeded.
```

You will now be able to run your desired CLI commands as outlined above. To see all possible functions as well as view the required inputs and expected outputs, add the `--help` flag to each command or check out the above documentation.

## Project Management

```bash
datature project - auth/list/select project from saved project.

positional arguments:
  {auth,select,list,help}
    auth                Authenticate and save the project.
    select              Select the project from saved projects.
    list                List the saved projects.
    help                Show this help message and exit.

optional arguments:
  -h, --help            show this help message and exit
```

## Asset Management

```bash
datature asset - upload/group assets.

positional arguments:
  {upload,group,help}
    upload             Bulk update assets.
    group              List assets group details.
    help               Show this help message and exit.

optional arguments:
  -h, --help           show this help message and exit
```

## Annotation Management

```bash
datature annotation - upload/download annotations.

positional arguments:
  {upload,download,help}
    upload              Bulk upload annotations from file.
    download            Bulk download annotations to file.
    help                Show this help message and exit.

optional arguments:
  -h, --help            show this help message and exit
```

## Artifact Management

```bash
datature artifact - download artifact models.

positional arguments:
  {download,help}
    download       Download artifact model.
    help           Show this help message and exit.

optional arguments:
  -h, --help       show this help message and exit
```
