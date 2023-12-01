import os
from zipfile import ZipFile

import cv2
import datature
import matplotlib.pyplot as plt
import numpy as np
import wget
from ultralytics import YOLO
from yolov8_cam.eigen_cam import EigenCAM
from yolov8_cam.utils.image import show_cam_on_image

model = YOLO("yolov8m.pt")

img = cv2.imread('images/bird-dog-cat.jpg')
img = cv2.resize(img, (320, 320))
rgb_img = img.copy()
img = np.float32(img) / 255
#plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
#plt.show()
target_layers = [model.model.model[-3]]

cam = EigenCAM(model, target_layers,task='od')
grayscale_cam = cam(
    rgb_img,
    eigen_smooth=True,
    principal_comp=[0],
)[0, :, :]
#cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

cam = EigenCAM(model, target_layers,task='od')
grayscale_cam = cam(
    rgb_img,
    eigen_smooth=True,
    principal_comp=[1],
)
print("grayscale_cam", grayscale_cam.shape)
cam_image = show_cam_on_image(img, grayscale_cam[0,:,:,0], use_rgb=True)

plt.imshow(cam_image)
plt.show()
