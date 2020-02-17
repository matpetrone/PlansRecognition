import json
import cv2
import matplotlib.pyplot as plt
import os

class Dataset():
    def __init__(self, img_dir='Flo2PlanAll-Seg'):
        with open('Flo2PlanAll-Seg/annotations/flo2plan_instances_final.json') as json_file:
            self.json_file = json_file
        self.dir_img = os.path.join(os.path.abspath(img_dir), 'images')

    def __getitem__(self, item):
        img = cv2.imread(os.path.join(self.dir_img,item))

        return (img, annotation)