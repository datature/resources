"""Adapted from https://github.com/rigvedrs/YOLO-V8-CAM"""
import numpy as np


def get_2d_projection(activation_batch, principal_comp=[0]):
    # TBD: use pytorch batch svd implementation
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = (
            (activations).reshape(activations.shape[0], -1).transpose()
        )
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - reshaped_activations.mean(
            axis=0
        )
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)

        projection = reshaped_activations @ VT[principal_comp, :].T
        projection = projection.reshape(list(activations.shape[1:])+[len(principal_comp)])
        projections.append(projection)
    return np.float32(projections)
