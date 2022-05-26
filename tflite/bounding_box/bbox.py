### Import all dependencies 
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" ## Uncomment to use GPU

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib.pyplot import imshow
import argparse


## Prepare utility functions
def load_label_map(label_map_path):
    """Reads label map in the format of .pbtxt and parse into dictionary

    Args:
      label_map_path: the file path to the label_map

    Returns:
      dictionary with the format of {label_index: {'id': label_index, 'name': label_name}}
    """
    label_map = {}

    with open(label_map_path, "r") as label_file:
        for line in label_file:
            if "id" in line:
                label_index = int(line.split(":")[-1])
                label_name = next(label_file).split(":")[-1].strip().strip("'")
                label_map[label_index] = {
                    "id": label_index,
                    "name": label_name,
                }

    return label_map

def args_parser():
    parser = argparse.ArgumentParser(
        description="Datature Open Source Prediction Script"
    )
    parser.add_argument(
        "--INPUT", help="Path to folder that contains input images", default="./input/"
    )
    parser.add_argument(
        "--OUTPUT", help="Path to folder to store predicted images", default="./output/"
    )
    parser.add_argument(
        "--SIZE", help="Size of image to load into model", default="640,640"
    )
    parser.add_argument(
        "--THRESHOLD", help="Prediction confidence threshold", default=0.7
    )
    parser.add_argument(
        "--MODEL", help="Path to tensorflowlite model", default="./tf.lite"
    )
    parser.add_argument(
        "--LABEL", help="Path to tensorflow label map", default="./label_map.pbtxt"
    )
    return parser.parse_args()


def main():
    args = args_parser()
    args.SIZE = args.SIZE.split(',')
    args.SIZE[0] = int(args.SIZE[0])
    args.SIZE[1] = int(args.SIZE[1])
    args.THRESHOLD = float(args.THRESHOLD)

    ### Load model into interpreter
    interpreter = tf.lite.Interpreter(model_path=args.MODEL)

    ### Visualize details of input & output layers
    ## Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    ## Print output details.
    for idx, each_out in enumerate(output_details):
        print("{}: {}\n".format(idx,each_out))

    ### Redefine expected image size of input layer
    ## Current size is set to accept SIZE[0]xSIZE[1] by 320 image
    interpreter.resize_tensor_input(input_details[0]['index'], (1, args.SIZE[0], args.SIZE[1], 3))
    interpreter.allocate_tensors()


    ## Run prediction on each image
    for each_image in os.listdir(args.INPUT):
        print("processing "+each_image)
        ### Load image into numpy array and resize accordingly
        img = cv2.imread(args.INPUT+'/'+each_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origi_shape = img.shape
        new_img = cv2.resize(img, (args.SIZE[0], args.SIZE[1]))
        
        ### Execute prediction

        interpreter.set_tensor(input_details[0]['index'], [new_img.astype(np.uint8)])
        interpreter.invoke()
        

        ### Extract data and filter against desired threshold
        for each_layer in output_details:
            if each_layer["name"] == "StatefulPartitionedCall:4":
                scores = np.squeeze(interpreter.get_tensor(each_layer['index']))
                
            if each_layer["name"] == "StatefulPartitionedCall:1":
                bboxes = np.squeeze(interpreter.get_tensor(each_layer['index']))
                
            if each_layer["name"] == "StatefulPartitionedCall:2":
                classes = np.squeeze(interpreter.get_tensor(each_layer['index']))

        ## Current desired confidence threshold is set at 0.7
        indexes = np.where(scores > args.THRESHOLD)

        ## Filter all prediction data above threshold
        scores = scores[indexes]
        bboxes = bboxes[indexes]
        classes = classes[indexes]


        ### Load label map and color map
        category_index = load_label_map(args.LABEL)

        color_map = {}
        for each_class in range(len(category_index)):
            color_map[each_class] = [
                int(i) for i in np.random.choice(range(256), size=3)
            ]

        ### Draw mask, bounding box, label onto the original image
        image_origi = np.array(img)

        if len(bboxes) != 0:
            for idx, each_bbox in enumerate(bboxes):
                color = color_map.get(classes[idx] - 1)

                ## Draw bounding box
                cv2.rectangle(
                    image_origi,
                    (
                        int(each_bbox[1] * origi_shape[1]),
                        int(each_bbox[0] * origi_shape[0]),
                    ),
                    (
                        int(each_bbox[3] * origi_shape[1]),
                        int(each_bbox[2] * origi_shape[0]),
                    ),
                    color,
                    2,
                )

                ## Draw label background
                cv2.rectangle(
                    image_origi,
                    (
                        int(each_bbox[1] * origi_shape[1]),
                        int(each_bbox[2] * origi_shape[0]),
                    ),
                    (
                        int(each_bbox[3] * origi_shape[1]),
                        int(each_bbox[2] * origi_shape[0] + 15),
                    ),
                    color,
                    -1,
                )

                ## Insert label class & score
                cv2.putText(
                    image_origi,
                    "Class: {}, Score: {}".format(
                        str(category_index[classes[idx]]["name"]),
                        str(round(scores[idx], 2)),
                    ),
                    (
                        int(each_bbox[1] * origi_shape[1]),
                        int(each_bbox[2] * origi_shape[0] + 10),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        ## Save predicted image
        filename = os.path.basename(each_image)
        image_predict = Image.fromarray(image_origi)
        image_predict.save(os.path.join(args.OUTPUT, filename))

        print(
            "Saving predicted images to {}...".format(
                os.path.join(args.OUTPUT, filename)
            )
        )


if __name__ == "__main__":
    main()
