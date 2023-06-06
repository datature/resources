# Datature SDK Guides

This section contains guides for using our SDK in Python and via the command line.

## Getting Started

### Prerequisites

The SDK is publicly available on [PyPI](https://pypi.org/project/datature/). They can be installed by running the following command:

```bash
pip install -U datature
```

### [Python SDK](./python/python-sdk-guide.ipynb)

To verify that the SDK is installed correctly, run the following command:

```bash
python -c "import datature"
```

If no error is thrown, the SDK is installed correctly.

### [Command Line Interface](#)

To verify that the CLI is installed correctly, run the following command:

```bash
datature --help
```

You should see a help message similar to the one below:

```bash
usage: datature [-h] {project,asset,annotation,artifact} ...

Command line tool to create/upload/download datasets on datature nexus.

positional arguments:
  {project,asset,annotation,artifact}
    project             Project management.
    asset               Asset management.
    annotation          Annotation management.
    artifact            Artifact management.

optional arguments:
  -h, --help            show this help message and exit
```
