# Copyright (c) OpenMMLab. All rights reserved.
"""
conda activate pyskl
python stgcn_yolov8.py input/test.avi output/output.mp4 \
    --config configs/stgcn++/stgcn++_ntu60_bioniks/j.py \
    --checkpoint stgcn_model.pth \
    --pose-checkpoint /home/datature/PNH/yolov8Pose/runs/pose/nturgbd/weights/best.pt \
    --skeleton /home/datature/PNH/yolov8Pose/cfg/nturgbd_skeleton.txt \
    --det-score-thr 0.7 \
    --label-map tools/data/label_map/sit_stand.txt
"""
import argparse
import cv2
import mmcv
import numpy as np
import os
import os.path as osp
import shutil
import torch
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors
import json

from pyskl.apis import inference_recognizer, init_recognizer

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (0, 0, 0)  # BGR, black
THICKNESS = 2
LINETYPE = 1


def parse_args():
    parser = argparse.ArgumentParser(description='PoseC3D demo')
    parser.add_argument('video', help='video file/url')
    parser.add_argument('out_filename', help='output filename')
    parser.add_argument(
        '--config',
        default='configs/posec3d/slowonly_r50_ntu120_xsub/joint.py',
        help='skeleton action recognition config file path')
    parser.add_argument(
        '--checkpoint',
        default=
        'https://download.openmmlab.com/mmaction/pyskl/ckpt/posec3d/slowonly_r50_ntu120_xsub/joint.pth',
        help='skeleton action recognition checkpoint file/url')
    parser.add_argument(
        '--pose-checkpoint',
        default=
        'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth',
        help='human pose estimation checkpoint file/url')
    parser.add_argument('--det-score-thr',
                        type=float,
                        default=0.7,
                        help='the threshold of human detection score')
    parser.add_argument('--label-map',
                        default='tools/data/label_map/nturgbd_120.txt',
                        help='label map file')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='CPU/CUDA device option')
    parser.add_argument(
        '--skeleton',
        type=str,
        required=True,
        help='skeleton file path for rendering edges during visualization')
    args = parser.parse_args()
    return args


def frame_extraction(video_path):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1

        flag, frame = vid.read()
    return frame_paths, frames


def pose_inference(args, frame_paths):
    model = YOLO(args.pose_checkpoint)
    ret = []
    predictions_for_viz = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f in frame_paths:
        detections_output = model.predict(source=f,
                                          conf=args.det_score_thr,
                                          imgsz=640,
                                          iou=0.8,
                                          task='pose',
                                          verbose=False)[0]
        boxes = detections_output.boxes.xyxy.cpu().numpy(
        ) if detections_output.boxes.xyxy is not None else []
        classes = detections_output.boxes.cls.cpu().numpy(
        ) if detections_output.boxes.cls is not None else []
        box_scores = detections_output.boxes.conf.cpu().numpy(
        ) if detections_output.boxes.conf is not None else []
        keypoints = detections_output.keypoints.xy.cpu().numpy(
        )[0] if detections_output.keypoints.xy is not None else []
        normalized_keypoints = detections_output.keypoints.xyn.cpu().numpy(
        )[0] if detections_output.keypoints.xyn is not None else []
        keypoint_scores = detections_output.keypoints.conf.cpu().numpy(
        )[0] if detections_output.keypoints.conf is not None else []

        predictions = {
            "boxes": boxes,
            "classes": classes,
            "box_scores": box_scores,
            "keypoints": keypoints,
            "normalized_keypoints": normalized_keypoints,
            "keypoint_scores": keypoint_scores
        }
        predictions_for_viz.append(predictions)

        if len(boxes) == 0:
            ret.append([])
            prog_bar.update()
            continue

        # merge keypoints and keypoint_scores
        if len(keypoints) > 0:
            keypoints = np.concatenate([keypoints, keypoint_scores[..., None]],
                                       axis=1)
        pose = [{"bbox": boxes[0], "keypoints": keypoints}]
        ret.append(pose)

        prog_bar.update()
    return ret, predictions_for_viz


def dist_ske(ske1, ske2):
    dist = np.linalg.norm(ske1[:, :2] - ske2[:, :2], axis=1) * 2
    diff = np.abs(ske1[:, 2] - ske2[:, 2])
    return np.sum(np.maximum(dist, diff))


def pose_tracking(pose_results, max_tracks=2, thre=30):
    tracks, num_tracks = [], 0
    num_joints = None
    for idx, poses in enumerate(pose_results):
        if len(poses) == 0:
            continue
        if num_joints is None:
            num_joints = poses[0].shape[0]
        track_proposals = [t for t in tracks if t['data'][-1][0] > idx - thre]
        n, m = len(track_proposals), len(poses)
        scores = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                scores[i][j] = dist_ske(track_proposals[i]['data'][-1][1],
                                        poses[j])

        row, col = linear_sum_assignment(scores)
        for r, c in zip(row, col):
            track_proposals[r]['data'].append((idx, poses[c]))
        if m > n:
            for j in range(m):
                if j not in col:
                    num_tracks += 1
                    new_track = dict(data=[])
                    new_track['track_id'] = num_tracks
                    new_track['data'] = [(idx, poses[j])]
                    tracks.append(new_track)
    tracks.sort(key=lambda x: -len(x['data']))
    result = np.zeros((max_tracks, len(pose_results), num_joints, 3),
                      dtype=np.float16)
    for i, track in enumerate(tracks[:max_tracks]):
        for item in track['data']:
            idx, pose = item
            result[i, idx] = pose
    return result[..., :2], result[..., 2]


def draw_bbox(image, boxes, classes, scores):
    """Draw bounding boxes on image.

    Args:
        image (np.ndarray): Image to draw bounding boxes on.
        boxes (np.ndarray): Bounding boxes.
        classes (np.ndarray): Classes of bounding boxes.
        scores (np.ndarray): Confidence scores of bounding boxes.

    Returns:
        image (np.ndarray): Image with bounding boxes drawn on.
    """
    for each_bbox, each_class, each_score in zip(boxes, classes, scores):
        cv2.rectangle(image, (int(each_bbox[0]), int(each_bbox[1])),
                      (int(each_bbox[2]), int(each_bbox[3])), [0, 0, 0], 2)
        cv2.rectangle(image, (int(each_bbox[0]), int(each_bbox[3] - 15)),
                      (int(each_bbox[2]), int(each_bbox[3])), [0, 0, 0], -1)
        cv2.putText(
            image, f"Class: {int(each_class)}, "
            f"Score: {str(round(each_score, 2))}",
            (int(each_bbox[0] + 5), int(each_bbox[3] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, [255, 255, 255], 1, cv2.LINE_AA)
    return image


def draw_keypoints(image, keypoints, scores, skeleton):
    """Draw keypoints and skeleton on image.

    Args:
        image (np.ndarray): Image to draw keypoints on.
        keypoints (np.ndarray): Keypoints.
        scores (np.ndarray): Confidence scores of keypoints.
        skeleton (list[list[int, int]]): Skeleton of keypoints.

    Returns:
        image (np.ndarray): Image with keypoints and skeleton drawn on.
    """
    height, width, _ = image.shape
    nkpt, ndim = keypoints.shape
    colors = Colors()
    limb_color = colors.pose_palette[[
        16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 9, 9, 5, 5, 5, 5, 5, 5, 5, 5, 16, 9,
        9, 9, 9
    ]]
    for i, sk in enumerate(skeleton):
        pos1 = (int(keypoints[(sk[0] - 1), 0]), int(keypoints[(sk[0] - 1), 1]))
        pos2 = (int(keypoints[(sk[1] - 1), 0]), int(keypoints[(sk[1] - 1), 1]))
        if ndim == 3:
            conf1 = keypoints[(sk[0] - 1), 2]
            conf2 = keypoints[(sk[1] - 1), 2]
            if conf1 < 0.5 or conf2 < 0.5:
                continue
        if pos1[0] % width == 0 or pos1[1] % height == 0 or pos1[
                0] < 0 or pos1[1] < 0:
            continue
        if pos2[0] % width == 0 or pos2[1] % height == 0 or pos2[
                0] < 0 or pos2[1] < 0:
            continue
        cv2.line(image,
                 pos1,
                 pos2, [int(x) for x in limb_color[i]],
                 thickness=2,
                 lineType=cv2.LINE_AA)
    for i, (each_kp, each_score) in enumerate(zip(keypoints, scores)):
        color_k = colors(0)
        x_coord, y_coord = each_kp[0], each_kp[1]
        if x_coord % width != 0 and y_coord % height != 0:
            if len(each_kp) == 3:
                conf = each_kp[2]
                if conf < 0.5:
                    continue
            cv2.circle(image, (int(x_coord), int(y_coord)),
                       2,
                       color_k,
                       -1,
                       lineType=cv2.LINE_AA)
            cv2.putText(image, str(round(each_score, 2)),
                        (int(x_coord) - 10, int(y_coord) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, [0, 0, 0], 1,
                        cv2.LINE_AA)
    return image


def visualize(image, detections_output, skeleton):
    """Visualize predictions by drawing bounding boxes and keypoints on image.

    Args:
        image (np.ndarray): Image to visualize.
        detections_output (dict): Output from predict function.
        skeleton (list[list[int, int]]): Skeleton of keypoints.
        output_path (str): Path to save visualized image.

    Returns:
        image (np.ndarray):
            Image with bounding boxes and keypoints drawn.
    """
    image = draw_bbox(image, detections_output["boxes"],
                      detections_output["classes"],
                      detections_output["box_scores"])
    image = draw_keypoints(image, detections_output["keypoints"],
                           detections_output["keypoint_scores"], skeleton)
    return image


def main():
    args = parse_args()

    if args.skeleton:
        with open(args.skeleton, "r", encoding="utf-8") as skeleton_file:
            data = skeleton_file.read()
            skeleton = json.loads(data)
    else:
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    frame_paths, original_frames = frame_extraction(args.video)
    num_frame = len(frame_paths)
    h, w, _ = original_frames[0].shape

    config = mmcv.Config.fromfile(args.config)
    config.data.test.pipeline = [
        x for x in config.data.test.pipeline if x['type'] != 'DecompressPose'
    ]
    # Are we using GCN for Infernece?
    GCN_flag = 'GCN' in config.model.type
    GCN_nperson = None
    if GCN_flag:
        format_op = [
            op for op in config.data.test.pipeline
            if op['type'] == 'FormatGCNInput'
        ][0]
        # We will set the default value of GCN_nperson to 2, which is
        # the default arg of FormatGCNInput
        GCN_nperson = format_op.get('num_person', 2)

    model = init_recognizer(config, args.checkpoint, args.device)

    # Load label_map
    label_map = [x.strip() for x in open(args.label_map).readlines()]

    # Get Human detection results
    pose_results, predictions_for_viz = pose_inference(args, frame_paths)
    torch.cuda.empty_cache()

    fake_anno = dict(frame_dir='',
                     label=-1,
                     img_shape=(h, w),
                     original_shape=(h, w),
                     start_index=0,
                     modality='Pose',
                     total_frames=num_frame)

    # We will keep at most `GCN_nperson` persons per frame.
    tracking_inputs = [[pose['keypoints'] for pose in poses]
                       for poses in pose_results]
    keypoint, keypoint_score = pose_tracking(tracking_inputs,
                                             max_tracks=GCN_nperson)
    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score

    results = inference_recognizer(model, fake_anno)
    action_label = label_map[results[0][0]]
    print(f'\nAction: {action_label}')

    vis_frames = [cv2.imread(f) for f in frame_paths]
    for frame, prediction in zip(vis_frames, predictions_for_viz):
        visualize(frame, prediction, skeleton)
        cv2.rectangle(frame, (0, 0), (w, 50), (255, 255, 255), -1)
        cv2.putText(frame, action_label, (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    vid.write_videofile(args.out_filename, remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)


if __name__ == '__main__':
    main()
