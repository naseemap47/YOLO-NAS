from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from IPython.display import clear_output
from super_gradients.training import Trainer
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training import training_hyperparams
import torch
import cv2
import random
import numpy as np
import time


dataset_params = {
    'data_dir':'Data',
    'train_images_dir':'train/images',
    'train_labels_dir':'train/labels',
    'val_images_dir':'valid/images',
    'val_labels_dir':'valid/labels',
    'test_images_dir':'test/images',
    'test_labels_dir':'test/labels',
    'classes': ['Paper', 'Rock', 'Scissors']
}

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

model = models.get('yolo_nas_s',
                        num_classes=len(dataset_params['classes']),
                        checkpoint_path="checkpoints/my_first_yolonas_run/ckpt_best.pth")

model = model.to("cuda" if torch.cuda.is_available() else "cpu")
# class_names = open('classes.txt', 'r').read().splitlines()
class_names = dataset_params['classes']
print('Class Names: ', class_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

# http://159.130.70.206/mjpg/video.mjpg
cap = cv2.VideoCapture(0)
p_time = 0
while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Failed to read...')
        break
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preds = next(model.predict(img_rgb)._images_prediction_lst)
    # class_names_ = preds.class_names
    # print(class_names_)
    dp = preds.prediction
    bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)
    for box, cnf, cs in zip(bboxes, confs, labels):
        if cnf > 0.6:
            plot_one_box(box[:4], img, label=f'{class_names[cs]} {cnf:.3}', color=colors[cs])

    # FPS
    c_time = time.time()
    fps = 1/(c_time-p_time)
    p_time = c_time
    cv2.putText(
        img, f'FPS: {fps:.3}', (50, 60),
        cv2.FONT_HERSHEY_PLAIN, 2, 
        (0, 255, 0), 2
    )

    k = cv2.waitKey(1)
    cv2.imshow('img', img)
    if k == ord('q'):
        break

cap.release()
# if save:
#     out_vid.release()
cv2.destroyAllWindows()