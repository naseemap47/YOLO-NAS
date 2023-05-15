from super_gradients.training import models
from super_gradients.common.object_names import Models
import cv2
import numpy as np
import random
import torch


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
# colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]
img = cv2.imread('dog.jpeg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

preds = next(model.predict(img_rgb)._images_prediction_lst)
class_names = preds.class_names
dp = preds.prediction
bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)
for box, cnf, cs in zip(bboxes, confs, labels):
    plot_one_box(box[:4], img, label=class_names[cs])

cv2.imshow('img', img)
cv2.waitKey(0)