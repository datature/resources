#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
  ████
██    ██   Datature
  ██  ██   Powering Breakthrough AI
    ██
​
@File    :   annotation_maker.py
@Author  :   Leonard So
@Version :   1.0
@Contact :   hello@datature.io
@License :   Apache License 2.0
@Desc    :   Automatic bounding box annotation creation for whole image.
"""
import os
import csv
import argparse
from PIL import Image


def create_annotation_file(image_dir):
    """
    Creates CSV Four Corner - Bounding Box annotation file to upload to
    the Nexus platform. This should be used on local storage of images.
    The file structure should be as follows (according to your current
    tags):
    ├── OK
    │   ├── img1.jpg
    │   ├── ...
    ├── Flashing
    │   ├── img2.jpg
    │   ├── ...
    ├── ShortMould
    │   ├── img3.jpg
    │   ├── ...
    ├── Flashing_ShortMould
    │   ├── img4.jpg
    │   ├── ...
    Requires Pillow to run, so please install in virtual environment.
    Args:
      image_dir: string for folder path containing the above folder
      structure. The corresponding csv file will be saved in the
      folder as annotations.csv. Once it has been created, you can
      upload the annotation with the format as CSV Four Corner -
      Bounding Box.
    """
    f = open(image_dir + "/annotations.csv", "w")
    writer = csv.writer(f)
    header = ["filename", "xmin", "ymin", "xmax", "ymax", "label"]
    writer.writerow(header)
    acceptable_ext = [".png", ".jpg", ".jpeg"]
    for subdir, _, files in os.walk(image_dir):
        label = subdir.split("/")[-1]
        for file in files:
            filename, ext = os.path.splitext(file)
            if ext in acceptable_ext:
                width, height = Image.open(os.path.join(subdir, file)).size
                writer.writerow([filename, 0, 0, width, height, label])


def args_parser():
    """
    Creates arg parser for command line input.
    """
    parser = argparse.ArgumentParser(
        description="Datature Open Source Prediction Script"
    )
    parser.add_argument(
        "--dataset",
        help="Path to folder that contains image dataset",
        required=True,
    )
    return parser.parse_args()


if __name__ == """__main__""":
    args = args_parser()
    if os.path.exists(args.dataset) is False:
        raise Exception("Folder Path Does Not Exist")
    create_annotation_file(args.dataset)
