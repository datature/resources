"""Adapted from https://github.com/rigvedrs/YOLO-V8-CAM"""
from typing import Callable, List, Tuple

import numpy as np
import torch
import ttach as tta
from yolov8_cam.activations_and_gradients import ActivationsAndGradients
from yolov8_cam.utils.image import scale_cam_image
from yolov8_cam.utils.model_targets import ClassifierOutputTarget
from yolov8_cam.utils.svd_on_activations import get_2d_projection


class BaseCAM:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        task: str = "od",
        reshape_transform: Callable = None,
        compute_input_gradient: bool = False,
        uses_gradients: bool = True,
    ) -> None:
        self.model = model
        self.target_layers = target_layers
        self.task = task
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform
        )

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(
        self,
        input_tensor: np.array,
        target_layers: List[torch.nn.Module],
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
    ) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(
        self,
        input_tensor: np.array,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        weights = self.get_cam_weights(
            input_tensor, target_layer, targets, activations, grads
        )
        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(
        self,
        input_tensor: np.array,
        targets: List[torch.nn.Module],
        eigen_smooth: bool = False,
        principal_comp: List[int] = [0],
    ) -> np.ndarray:
        outputs = self.activations_and_grads(input_tensor)
        if targets is None:
            #             target_categories = np.argmax(outputs[0].cpu().data.numpy(), axis=-1)
            if self.task == "od":
                target_categories = outputs[0].boxes.cls
            elif self.task == "cls":
                target_categories = outputs[0].probs.top5
            else:
                print("not ok")
            targets = [
                ClassifierOutputTarget(category)
                for category in target_categories
            ]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum(
                [target(output) for target, output in zip(targets, outputs)]
            )
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(
            input_tensor, targets, eigen_smooth, principal_comp=principal_comp,
        )
        return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(
        self, input_tensor: np.array
    ) -> Tuple[int, int]:
        width, height = np.shape(input_tensor)[0], np.shape(input_tensor)[1]
        return width, height

    def compute_cam_per_layer(
        self,
        input_tensor: np.array,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
        principal_comp: List[int] = [0],
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(layer_activations, principal_comp=principal_comp)
            cam = np.maximum(cam, 0)
            scaled = scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(
        self, cam_per_target_layer: np.ndarray
    ) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result)

    def forward_augmentation_smoothing(
        self,
        input_tensor: np.array,
        targets: List[torch.nn.Module],
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor, targets, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(
        self,
        input_tensor: np.array,
        targets: List[torch.nn.Module] = None,
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
        principal_comp: List[int] = [0],
    ) -> np.ndarray:
        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, targets, eigen_smooth
            )

        return self.forward(input_tensor, targets, eigen_smooth, principal_comp=principal_comp)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}"
            )
            return True
