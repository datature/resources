"""Adapted from https://github.com/rigvedrs/YOLO-V8-CAM"""
from yolov8_cam.base_cam import BaseCAM
from yolov8_cam.utils.svd_on_activations import get_2d_projection

# https://arxiv.org/abs/2008.00299


class EigenCAM(BaseCAM):

    def __init__(
            self,
            model,
            target_layers,
            task: str = 'od',  #use_cuda=False,
            reshape_transform=None):
        super().__init__(model,
                         target_layers,
                         task,
                         reshape_transform,
                         uses_gradients=False)

    def get_cam_image(self, activations):
        return get_2d_projection(activations)
