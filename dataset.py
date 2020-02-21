import json
import cv2
import matplotlib.pyplot as plt
import os
import random

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

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


register_coco_instances("plans", {}, "./Flo2PlanAll-Seg/annotations/flo2plan_instances_final.json", "./Flo2PlanAll-Seg/images")
plans_metadata = MetadataCatalog.get("plans")
dataset_dicts = DatasetCatalog.get("plans")

for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=plans_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow('Test',vis.get_image()[:, :, ::-1])
    cv2.waitKey(delay=0)