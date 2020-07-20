import numpy as np
import os
from matplotlib import pyplot as plt
from dataset.imagen import Imagen

class Dataset():

    def __init__(self,folder="examples"):
        self.folder= folder
        self.images = []
        for img in os.listdir(self.folder):
            if os.path.isfile(self.folder+"/"+img):
                self.images.append(img)
        print(str(len(self.images))+" images find in the directory")

    def show_some_images(self):
        x = 1
        for i in range(0,4):
            plt.subplot(2, 2, x)
            img_array=np.load(self.folder+"/"+self.images[i])
            plt.imshow(img_array, cmap='gray')
            x = x+1
        plt.show()

    def show_all(self):
        i = 1
        for img in range(0,15):
            f1 = plt.figure(i)
            i= i+1
            img_array=np.load(self.folder+"/"+self.images[img])
            plt.imshow(img_array, cmap='gray')
        plt.show()

    def clean_dataset(self,cadena="00001"):
        for img in self.images:
            #if(img.startswith(cadena)):
            img_array=np.load(self.folder+"/"+img)
            plt.imshow(img_array, cmap='gray')
            plt.show()
            value = input()
            if value=="1":
                np.save("examples/"+img, img_array)
                print("Guardado")

    def get_img(self,index=0):
        return Imagen(self.folder+"/"+self.images[index],self.images[index])

    def get_size(self):
        return len(self.images)
