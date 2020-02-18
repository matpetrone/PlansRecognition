from dataset import Dataset
import cv2
import matplotlib.pyplot as plt
import numpy as np

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

def drawContours(img, annotations):
    thickness = 2
    for a in annotations:
        contours = []
        arr=np.array([])
        for s in range(0,len(a),2):
            ar =np.array([int(a['segmentation'][0][s]), int(a['segmentation'][0][s+1])])
            arr = np.append(arr,ar)
        print (len(a['segmentation'][0])/2)
        arr.reshape(int(len(a['segmentation'][0])/2),1,2)
        contours.append(arr)




            #contours.append((int(a['segmentation'][0][s]),int(a['segmentation'][0][s+1])))


        #image = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    #plt.imshow(image)
    #plt.show()
    print (contours)

dataset = Dataset()
(img, ann) = dataset['001.png']
drawBoxes(img, ann)
drawContours(img,ann)

