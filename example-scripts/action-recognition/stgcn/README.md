# Action Recognition with ST-GCN++ and YOLOv8-Pose

This folder contains the code and resources used in the guide [How to Perform Action Recognition with YOLOv8 and ST-GCN++](https://www.datature.io/blog/) on a custom-trained models on the sit-stand subset of the [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) dataset.

## Install Prerequisites

```bash
pip install -r requirements.txt
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
# This command runs well with conda 22.9.0,
# if you are running an early conda version and face some errors,
# try to update your conda first
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```

## Train Custom STGCN++ Model

### Prepare Data & Configs (Optional)

<details>

<summary>Click to expand</summary>

The sit-stand subset of the NTU-RGB+D dataset has been preprocessed and conveniently included as a pickle file (`sit_stand.pkl`) that can directly be used for training. If you wish to use a different subset or a custom dataset, do check out this [guide](https://github.com/kennymckormick/pyskl/blob/main/tools/data/README.md) to generate your own pickle file.

The training configuration file will also need to be updated to reflect the path to the updated pickle file. Do note that the PYSKL library does not provide a convenient way to define a custom skeleton as part of the training configuration file or adding arguments to the training execution command. Instead, you will need to add your custom skeleton directly into the PYSKL code in [`pyskl/pyskl/utils/graph.py`](pyskl/pyskl/utils/graph.py) (function `get_layout()` on line 97) as shown in the following code snippet:

```python
self.num_node = NUM_KEYPOINTS
self.inward = [(kp1, kp2) pairs of adjacent keypoints]
self.center = CENTER_KEYPOINT_ID
```

You will also need to add the adjacency matrix of your custom skeleton into [`pyskl/pyskl/datasets/pipelines/pose_related.py`](pyskl/pyskl/datasets/pipelines/pose_related.py) (function `__init__()` of class `JointToBone` on line 295) as shown in the following code snippet:

```python
self.pairs = ((kp1, kp2) pairs of adjacent keypoints)
```

</details>

### Run Training

```bash
bash pyskl/tools/dist_train.sh configs/j.py <NUM_GPU> \
    --validate \
    --test-last \
    --test-best
```

## Test Custom STGCN++ Model

```bash
python stgcn_yolov8.py input/sample.avi output/output.mp4 \
    --config configs/j.py \
    --checkpoint test/stgcn_model.pth \
    --pose-checkpoint test/yolov8s-pose.pt \
    --skeleton test/nturgbd_skeleton.txt \
    --det-score-thr 0.7 \
    --label-map test/sit_stand.txt
```
