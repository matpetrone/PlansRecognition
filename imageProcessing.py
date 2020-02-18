from dataset import Dataset
import cv2
import matplotlib.pyplot as plt

def drawBoxes(img, annotations):
    thickness = 2
    for a in annotations:
        x1 = int(a['bbox'][0])
        y1 = int(a['bbox'][1])
        x2 = int(x1+a['bbox'][2])
        y2 = int(y1+a['bbox'][3])
        image = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), thickness)
    plt.imshow(image)
    plt.show()

dataset = Dataset()
(img, ann) = dataset['001.png']
drawBoxes(img, ann)
