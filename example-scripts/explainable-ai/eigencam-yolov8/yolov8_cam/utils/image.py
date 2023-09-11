"""Adapted from https://github.com/rigvedrs/YOLO-V8-CAM"""
import cv2
import numpy as np


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb:
        Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def scale_cam_image(cam, target_size=None):
    """Scales the cam image to be between 0 and 1."""
    result = []
    for img in cam:
        indiv_components = []
        for i in range(img.shape[2]):
            indiv_img = img[:,:,i]
            indiv_img = indiv_img - np.min(indiv_img)
            indiv_img = indiv_img / (1e-7 + np.max(indiv_img))
            if target_size is not None:
                indiv_img = cv2.resize(indiv_img, target_size)
            indiv_components.append(indiv_img)
        result.append(indiv_components)
    result = np.transpose(np.float32(result), axes=(0,2,3,1))
    return result
