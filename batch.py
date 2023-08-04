from super_gradients.training import models
import torch
import cv2
import random
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num", type=int, required=True,
                help="number of classes the model trained on")
ap.add_argument("-m", "--model", type=str, default='yolo_nas_s',
                choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'],
                help="Model type (eg: yolo_nas_s)")
ap.add_argument("-w", "--weight", type=str, required=True,
                help="path to trained model weight")
ap.add_argument("-s", "--source", nargs='+', default=[],
                help="paths to videos/cam-ids/RTSPs")
ap.add_argument("-c", "--conf", type=float, default=0.25,
                help="model prediction confidence (0<conf<1)")
ap.add_argument("--full", action='store_true',
                help="Enable full screen window")
args = vars(ap.parse_args())


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


# Load YOLO-NAS Model
model = models.get(
    args['model'],
    num_classes=args['num'], 
    checkpoint_path=args["weight"]
)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
class_names = model.predict(np.zeros((1,1,3)), conf=args['conf'])._images_prediction_lst[0].class_names
print('Class Names: ', class_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

# video cap for all sources
cap_list = []
for i in args['source']:
    if i.isnumeric():
        i = int(i)
    cap_temp = cv2.VideoCapture(i)
    cap_list.append(cap_temp)

if args['full']:
    for i in range(len(args['source'])):
        cv2.namedWindow(f'img{i}', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(f'img{i}', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    img_input_list = []
    img_list = []
    for id, cap in enumerate(cap_list):
        success, img = cap.read()
        if not success:
            print(f'[INFO] Failed to read {id}...')
            break
        else:
            img_list.append(img)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_input_list.append(img_rgb)
    
    preds = model.predict(img_input_list, conf=args['conf'])._images_prediction_lst
    for id, pred in enumerate(preds):
        # class_names = preds.class_names
        dp = pred.prediction
        bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)
        for box, cnf, cs in zip(bboxes, confs, labels):
            plot_one_box(box[:4], img_list[id], label=f'{class_names[int(cs)]} {cnf:.3}', color=colors[cs])

    for i in range(len(img_list)):
        cv2.imshow(f'img{i}', img_list[i])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
