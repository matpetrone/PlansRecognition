import json
import cv2
import matplotlib.pyplot as plt
import os

class Dataset():
    def __init__(self, img_dir='Flo2PlanAll-Seg'):
        with open('Flo2PlanAll-Seg/annotations/flo2plan_instances_final.json') as json_file:
            self.data = json.load(json_file)
        self.dir_img = os.path.join(os.path.abspath(img_dir), 'images')

    def __getitem__(self, item):
        img = cv2.imread(os.path.join(self.dir_img,item))
        annotations = []
        for i in self.data['images']:
            if i['file_name']==item:
                id = i['id']
        for a in self.data['annotations']:
            if a['image_id']==id:
                annotations.append(a)
        return (img, annotations)


dataset = Dataset()
list = dataset['001.png']

#print(list[1][0]['bbox'])
#print(list[1][0]['segmentation'])

