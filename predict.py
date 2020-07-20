import argparse
import os
import cv2
import imutils
import numpy as np
from dataset.model import Model
from dataset.main import Dataset

# Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=False, default="mask_rcnn_coco.h5", help="path to Mask R-CNN model weights pre-trained on COCO")
ap.add_argument("-l", "--labels", required=False, default="coco_labels.txt", help="path to class labels file")
ap.add_argument("-d", "--dataset", required=True, help="path to input image to apply Mask R-CNN to")
ap.add_argument("-s", "--size", required=False, help="size to the test the script, if not all the images are processed")
args = vars(ap.parse_args())

if (not os.path.isdir('masks/')):
	os.mkdir('masks/')

model = Model(args["weights"],args["labels"])
dataset = Dataset(folder=args["dataset"])
size = dataset.get_size()
if(args['size'] != None):
	size = int(args['size'])

for i in range(0,size):
	img = dataset.get_img(i)
	img.load()
	img.bgr2rgb()
	r = model.detect([img.img])[0]
	if((r["masks"]).shape[2]>0):
		mask = r["masks"][:, :, 0].astype(np.uint8)*255
		label = model.CLASS_NAMES[r["class_ids"][0]]
		#mask_try = mask.astype(bool)
		cv2.imwrite("masks/mask_"+label+"_"+str(r["scores"][0])+"_"+img.name+".jpeg",mask)
		mask_rgb = np.zeros(img.img.shape, np.uint8)
		#print(mask_rgb.shape)
		mask_rgb[:, :, 0] = mask
		#print(mask_rgb.shape)
		image_new = cv2.addWeighted(img.img, 0.4, mask_rgb, 0.6, 0)
		cv2.imwrite("masks/original_"+label+"_"+str(r["scores"][0])+"_"+img.name+".jpeg",image_new)
		#print(img.name)
