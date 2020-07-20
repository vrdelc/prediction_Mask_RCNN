import numpy as np
import cv2 as cv

class Imagen():

    def __init__(self,path,name):
        self.path= path
        self.name = name.split(".npy")[0]

    def load(self):
        self.img = np.load(self.path)

    def load_mask(self):
        self.img = cv.imread(self.path+"/"+self.name,0)

    def load_image(self):
        self.img = cv.imread(self.path)

    def preprocess(self):
        self.original = self.img.copy()
        self.blur()
        self.rgb2gray()
        self.denoise()

    def save(self,name):
        from PIL import Image
        im = Image.fromarray(self.img)
        im.save(name+".jpeg")

    def getImage(self):
        return self.img
        
    def bgr2rgb(self):
        self.img = cv.cvtColor(self.img, cv.COLOR_BGR2RGB)
