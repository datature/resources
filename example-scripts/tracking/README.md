# Tracking

This section contains example scripts for single and multi-object tracking. As a start, we provide sample scripts for several popular tracking algorithms:

- [BYTETrack](BYTETrack/bytetrack_basketball.ipynb)

BYTETrack keeps non-background low score boxes for a secondary association step between previous frame and next frame based on their similarities with tracklets. This helps to improve tracking consistency by keeping relevant bounding boxes which otherwise would have been discarded due to low confidence score (due to occlusion or appearance changes). The generic framework makes BYTETrack highly adaptable to any object detection (YOLO, RCNN) or instance association components (IoU, feature similarity).

<p align="center">
  <img alt="Tracker Comparison" src="https://uploads-ssl.webflow.com/62e939ff79009c74307c8d3e/64012caf1281d67f8651c943_636a149d733f01a3c9bf41a6_Tracking%2520Algorithm%2520Comparison.gif" width="90%">
</p>

- [DeepSORT](DeepSORT/DeepSORT.ipynb)

DeepSORT mainly uses the Kalman filter and the Hungarian algorithm for object tracking. Kalman filtering is used to predict the state of tracks in the previous frame in the current frame. The Hungarian algorithm associates the tracking frame tracks in the previous frame with the detection frame detections in the current frame, and performs tracks matching by calculating the cost matrix.

- [IoU](IoU/IoU.ipynb)

This method relies entirely on the detection results rather than the image itself. Intersection over union (IoU) is used to calculate the overlap rate between two frames. When IoU reaches the threshold, the two frames are considered to belong to the same track. Since this method relies solely on IoU, it assumes that every object is detected in every frame or that the "gap" in between is small and the distance between two detections is not too large, i.e. video frame rate is high. The IOU is calculated by: IOU(a, b) = (Area(a) Area(b)) (Area(a) Area(b))

- [Norfair](Norfair/Norfair.ipynb)

Norfair is a object tracking library based on DeepSORT algorithm.However, Norfair provides users with a high degree of customization, such as the distance function. The distance function we set here is to calculate the distance between the center points of the two boxes. Norfair also use the Kalman filter but it uses its own distance function instead of the Hungarian algorithm. And since Norfair does not use deep embedding like pure DeepSORT, it cannot be well matched again for occluded objects. But it will be faster than pure DeepSORT.

The jupyter notebooks can be run locally or on Google Colab as well.

The community is welcome to contribute other recent and state-of-the-art tracking algorithms to this repository.
