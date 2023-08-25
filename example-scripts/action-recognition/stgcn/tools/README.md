# NTU-RGB+D 2D Keypoint Conversion

There exists a pretrained ST-GCN model on the NTU-RGB+D dataset, which is trained on the 2D skeleton data. However, the labels are only compatible with HRNet, which uses COCO Pose (17 keypoints) format. The customizations in this repository are to utilize the original 25 keypoints to train a custom ST-GCN model. `convert_keypoints.py` modifies the pickle file that is used during ST-GCN training by replacing the keypoint labels.

The default data directories/files are assumed to be:

- Skeleton data: `data/nturgb+d_skeletons`
- Missing skeletons: `data/nturgb+d_missing_skeletons.txt`

## Usage

```bash
python convert_keypoints.py
```
