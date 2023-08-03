from super_gradients.training import models
import torch
import cv2
import random
import numpy as np
import time
import argparse
import os


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


def get_bbox(img):
    preds = model.predict(img, conf=args['conf'])._images_prediction_lst[0]
    # class_names = preds.class_names
    dp = preds.prediction
    bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)
    for box, cnf, cs in zip(bboxes, confs, labels):
        plot_one_box(box[:4], img, label=f'{class_names[int(cs)]} {cnf:.3}', color=colors[cs])
    return labels, class_names


# Load YOLO-NAS Model
model = models.get(
    'yolo_nas_s',
    pretrained_weights='coco'
)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
class_names = model.predict(np.zeros((1,1,3)), conf=0.25)._images_prediction_lst[0].class_names
print('Class Names: ', class_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

cap1 = cv2.VideoCapture('https://csea-me-webcam.cse.umn.edu/mjpg/video.mjpg?timestamp=1443034719346')
cap2 = cv2.VideoCapture('http://view.dikemes.edu.gr/mjpg/video.mjpg')

while True:
    success1, img1 = cap1.read()
    if not success1:
        print('[INFO] Failed to read1...')
        break
    success2, img2 = cap2.read()
    if not success2:
        print('[INFO] Failed to read2...')
        break

    img = [img1, img2]
    img_rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img_rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    preds = model.predict([img_rgb1, img_rgb2], conf=0.25)._images_prediction_lst
    for id, pred in enumerate(preds):
        # class_names = preds.class_names
        dp = pred.prediction
        bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)
        for box, cnf, cs in zip(bboxes, confs, labels):
            plot_one_box(box[:4], img[id], label=f'{class_names[int(cs)]} {cnf:.3}', color=colors[cs])

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
