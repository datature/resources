import json
import pickle
import os
import math
import numpy as np
import wget

HEIGHT, WIDTH = 1080, 1920


def read_skeleton_file(filename):
    with open(filename, 'r') as file:
        framecount = int(file.readline().strip())
        bodyinfo = []

        for _ in range(framecount):
            bodycount = int(file.readline().strip())
            frame_bodies = []

            for _ in range(bodycount):
                body = {}
                arrayint = list(map(float, file.readline().strip().split()))
                body['bodyID'] = int(arrayint[0])
                body['clipedEdges'] = int(arrayint[1])
                body['handLeftConfidence'] = int(arrayint[2])
                body['handLeftState'] = int(arrayint[3])
                body['handRightConfidence'] = int(arrayint[4])
                body['handRightState'] = int(arrayint[5])
                body['isResticted'] = int(arrayint[6])
                body['leanX'] = arrayint[7]
                body['leanY'] = arrayint[8]
                body['trackingState'] = int(arrayint[9])
                body['jointCount'] = int(file.readline().strip())
                joints = []

                for _ in range(body['jointCount']):
                    jointinfo = list(
                        map(float,
                            file.readline().strip().split()))
                    joint = {}
                    joint['x'] = jointinfo[0]
                    joint['y'] = jointinfo[1]
                    joint['z'] = jointinfo[2]
                    joint['depthX'] = jointinfo[3]
                    joint['depthY'] = jointinfo[4]
                    joint['colorX'] = jointinfo[5]
                    joint['colorY'] = jointinfo[6]
                    joint['orientationW'] = jointinfo[7]
                    joint['orientationX'] = jointinfo[8]
                    joint['orientationY'] = jointinfo[9]
                    joint['orientationZ'] = jointinfo[10]
                    joint['trackingState'] = jointinfo[11]
                    joints.append(joint)
                body['joints'] = joints
                frame_bodies.append(body)
            bodyinfo.append({'bodies': frame_bodies})
    return bodyinfo


def convert_keypoints(skeleton_filename, out_dir):
    bodyinfo = read_skeleton_file(skeleton_filename)
    keypoints_for_all_frames = []
    keypoints_for_pickle = []

    for f in range(len(bodyinfo)):
        keypoints = []
        for b in range(len(bodyinfo[f]['bodies'])):
            for j in range(25):
                joint = bodyinfo[f]['bodies'][b]['joints'][j]
                if math.isnan(float(joint['colorX'])) or math.isnan(
                        float(joint['colorY'])) or float(
                            joint['colorX']) <= 0.0 or float(
                                joint['colorY']) <= 0.0 or float(
                                    joint['colorX']) == math.inf or float(
                                        joint['colorY']) == math.inf or float(
                                            joint['colorX']) >= WIDTH or float(
                                                joint['colorY']) >= HEIGHT:
                    return None
                dx = float(joint['colorX'])
                dy = float(joint['colorY'])
                keypoints.append([dx, dy])

        keypoints_for_pickle.append(keypoints)
        keypoints_for_all_frames.append({
            "frame_id": f,
            "keypoints": keypoints
        })

    # dump to JSON file
    with open(
            os.path.join(
                out_dir,
                os.path.basename(skeleton_filename).split(".")[0] + ".json"),
            "w") as f:
        json.dump(keypoints_for_all_frames, f)

    # add one dimension to keypoints_for_pickle
    return np.expand_dims(np.array(keypoints_for_pickle, dtype=np.float16),
                          axis=0)


if __name__ == "__main__":
    skeleton_dir = "data/nturgb+d_skeletons"
    out_dir = "data/nturgb+d_keypoints"
    missing_skeleton_file = "data/sit_stand/missing_skeleton.txt"
    with open(missing_skeleton_file, "r") as f:
        missing_skeletons = f.read().splitlines()
    if not os.path.exists("data/ntu60_hrnet.pkl"):
        wget.download(
            "https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl",
            "data/ntu60_hrnet.pkl")
    pickle_file = open("data/ntu60_hrnet.pkl", "rb")
    data = pickle.load(pickle_file)

    new_annotations = []
    count = 0

    for file_path in sorted(os.listdir(skeleton_dir)):
        frame_name = file_path.split(".")[0]
        count += 1
        print(count, end="\r")
        try:
            if frame_name in missing_skeletons:
                continue

            skeleton_filename = os.path.join(skeleton_dir, file_path)
            keypoints = convert_keypoints(skeleton_filename, out_dir)

            if keypoints is None:
                with open(missing_skeleton_file, "a", encoding="utf-8") as f:
                    f.write(frame_name + "\n")
                continue

            ann = next((ann for ann in data['annotations']
                        if frame_name in ann['frame_dir']), None)

            if ann is None:
                continue

            new_annotations.append({
                "frame_dir":
                frame_name,
                "label":
                int(ann["label"]),
                "img_shape":
                ann["img_shape"],
                "original_shape":
                ann["original_shape"],
                "total_frames":
                keypoints.shape[1],
                "keypoint":
                keypoints,
                "keypoint_score":
                np.ones(keypoints.shape[:3], dtype=np.float16) * 0.9
            })
        except Exception as e:
            with open(missing_skeleton_file, "a", encoding="utf-8") as f:
                f.write(frame_name + "\n")
            continue

    data["annotations"] = new_annotations
    annotation_names = [ann["frame_dir"] for ann in data["annotations"]]

    for item in data['split']:
        # only keep entries containing A008 or A009
        data['split'][item] = [
            x for x in data['split'][item] if x in annotation_names
        ]
        # remove those entries with missing skeleton
        data['split'][item] = [
            x for x in data['split'][item] if x not in missing_skeletons
        ]

    print(len(data["annotations"]))
    print(data["annotations"][0])

    with open("data/ntu60_25kp.pkl", "wb") as f:
        pickle.dump(data, f)
