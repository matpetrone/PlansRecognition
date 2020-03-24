import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


register_coco_instances("plans_train", {}, "./Flo2PlanAll-Seg/annotations/flo2plan_instances_final.json", "./Flo2PlanAll-Seg/images")
#register_coco_instances("plans_test", {}, "./Flo2PlanAll-Seg/annotations/flo2plan_instances_final.json", "./Flo2PlanAll-Seg/images/test")
plans_metadata = MetadataCatalog.get("plans_train")
dataset_dicts = DatasetCatalog.get("plans_train")
#dataset_dicts_test = DatasetCatalog.get("plans_test")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("plans_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4  #prima =2
cfg.SOLVER.BASE_LR = 0.0005  # pick a good LR prima=0.00025 balloon , 0.02 nuts
cfg.SOLVER.MAX_ITER = 500    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 12  # only has one class (ballon)

cfg.INPUT.MIN_SIZE_TRAIN=(800,)
cfg.INPUT.MAX_SIZE_TRAIN=800
cfg.INPUT.MIN_SIZE_TEST=800
cfg.INPUT.MAX_SIZE_TEST=800

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model
cfg.DATASETS.TEST = ("plans_train",)
predictor = DefaultPredictor(cfg)


'''for i,d in enumerate(random.sample(dataset_dicts, 150)):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=plans_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    path= './test'
    #cv2.imshow("1",v.get_image()[:, :, ::-1])
    cv2.imwrite(os.path.join(path, '{}.jpg'.format(i+1)), v.get_image()[:, :, ::-1])
    #cv2.waitKey(delay=10000)'''

evaluator = COCOEvaluator("plans_train", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "plans_train")
inference_on_dataset(trainer.model, val_loader, evaluator)