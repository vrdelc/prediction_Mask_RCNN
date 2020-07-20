# Predict with Mask Mask_RCNN

## Configuration

Clone this repository

Install Mask RCNN as indicated in [github](https://github.com/matterport/Mask_RCNN)

Due to problem [AttributeError: module 'tensorflow' has no attribute 'log']( https://github.com/matterport/Mask_RCNN/issues/1797), install TensorFlow version 1.13.1 and Keras 2.1.0

You can download a trained model in [releases](https://github.com/matterport/Mask_RCNN/releases)

## How to use the script

Call predict.py script with this parameters:
* "-d", "--dataset", required=True, help="path to input image to apply Mask R-CNN to"
* "-w", "--weights", required=True, default="mask_rcnn_coco.h5", help="path to Mask R-CNN model weights pre-trained on COCO"
* "-l", "--labels", required=False, default="coco_labels.txt", help="path to class labels file"
* "-s", "--size", required=False, help="size to the test the script, if not all the images are processed"
