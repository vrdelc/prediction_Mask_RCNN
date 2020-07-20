import os
from mrcnn.config import Config
from mrcnn import model as modellib

class Model():

    def __init__(self,file_model, file_labels):
        self.CLASS_NAMES = open(file_labels).read().strip().split("\n")

        class SimpleConfig(Config):
        	NAME = "coco_inference"
        	GPU_COUNT = 1
        	IMAGES_PER_GPU = 1
        	NUM_CLASSES = len(self.CLASS_NAMES)-1 #remove first line

        config = SimpleConfig()

        print("[INFO] loading Mask R-CNN model...")
        self.model = modellib.MaskRCNN(mode="inference", config=config,	model_dir=os.getcwd())
        self.model.load_weights(file_model, by_name=True)

    def detect(self,images):
        """
        r =
        {
            "rois": array([startY, startX, endY, endX]),
            "class_ids": array(class_id),
            "scores":array([score]),
            "masks": array([mask])
        }
        """
        return self.model.detect(images, verbose=0)
