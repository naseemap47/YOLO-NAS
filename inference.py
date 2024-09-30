from super_gradients.training import models
import torch
import cv2
import random
import numpy as np
import time
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num", type=int, required=False,
                help="number of classes the model trained on")
ap.add_argument("-m", "--model", type=str, default='yolo_nas_s',
                choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'],
                help="Model type (eg: yolo_nas_s)")
ap.add_argument("-w", "--weight", type=str, required=True,
                help="path to trained model weight")
ap.add_argument("-s", "--source", type=str, required=True,
                help="video path/cam-id/RTSP")
ap.add_argument("-c", "--conf", type=float, default=0.25,
                help="model prediction confidence (0<conf<1)")
ap.add_argument("--save", action='store_true',
                help="Save video")
ap.add_argument("--hide", action='store_false',
                help="to hide inference window")
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


def get_bbox(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preds = model.predict(img_rgb, conf=args['conf'])._images_prediction_lst[0]
    # class_names = preds.class_names
    dp = preds.prediction
    bboxes, confs, labels = np.array(dp.bboxes_xyxy), dp.confidence, dp.labels.astype(int)
    for box, cnf, cs in zip(bboxes, confs, labels):
        plot_one_box(box[:4], img, label=f'{class_names[int(cs)]} {cnf:.3}', color=colors[cs])
    return labels


# Load COCO YOLO-NAS Model
if args["weight"] == "coco":
    model = models.get(args['model'], pretrained_weights="coco")
# Load YOLO-NAS Model
else:
    model = models.get(
        args['model'],
        num_classes=args['num'], 
        checkpoint_path=args["weight"]
    )

model = model.to("cuda" if torch.cuda.is_available() else "cpu")
class_names = model.predict(np.zeros((1,1,3)), conf=args['conf'])._images_prediction_lst[0].class_names
print('Class Names: ', class_names)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

# Global Timer
global_timer = time.time()

# Inference Image
if args['source'].endswith('.jpg') or args['source'].endswith('.jpeg') or args['source'].endswith('.png'):
    img = cv2.imread(args['source'])
    labels = get_bbox(img)
    # Timer
    print(f'[INFO] Completed in \033[1m{(time.time()-global_timer)/60} Minute\033[0m')
    
    if args['hide'] is False and len(labels)>0:
        pre_list = [class_names[int(x)] for x in labels]
        count_pred = {i:pre_list.count(i) for i in pre_list}
        print(f'Prediction: {count_pred}')
    
    # save Image
    if args['save'] or args['hide'] is False:
        os.makedirs(os.path.join('runs', 'detect'), exist_ok=True)
        path_save = os.path.join('runs', 'detect', os.path.split(args['source'])[1])
        cv2.imwrite(path_save, img)
        print(f"\033[1m[INFO] Saved Image: {path_save}\033[0m")

    # Hide video
    if args['hide']:
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('img', img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
    
    
# Reading Video/Cam/RTSP
else:
    video_path = args['source']
    if video_path.isnumeric():
        video_path = int(video_path)
    cap = cv2.VideoCapture(video_path)

    if args['hide'] is False:
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0

    # Get the width and height of the video - SAVE VIDEO.
    if args['save'] or args['hide'] is False:
        original_video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        os.makedirs(os.path.join('runs', 'detect'), exist_ok=True)
        if not str(video_path).isnumeric():
            path_save = os.path.join('runs', 'detect', os.path.split(video_path)[1])
        else:
            c = 0
            while True:
                if not os.path.exists(os.path.join('runs', 'detect', f'cam{c}.mp4')):
                    path_save = os.path.join('runs', 'detect', f'cam{c}.mp4')
                    break
                else:
                    c += 1
        out_vid = cv2.VideoWriter(path_save, 
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps, (original_video_width, original_video_height))

    p_time = 0
    if args['hide']:
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('img', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        success, img = cap.read()
        if not success:
            print('[INFO] Failed to read...')
            break
        
        labels = get_bbox(img)
        if args['hide'] is False and len(labels)>0:
            frame_count += 1
            pre_list = [class_names[int(x)] for x in labels]
            count_pred = {i:pre_list.count(i) for i in pre_list}
            print(f'Frames Completed: {frame_count}/{length} Prediction: {count_pred}')
            
        # FPS
        c_time = time.time()
        fps = 1/(c_time-p_time)
        p_time = c_time
        cv2.putText(
            img, f'FPS: {fps:.3}', (50, 60),
            cv2.FONT_HERSHEY_PLAIN, 2, 
            (0, 255, 0), 2
        )

        # Write Video
        if args['save'] or args['hide'] is False:
            out_vid.write(img)

        # Hide video
        if args['hide']:
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Timer
    print(f'[INFO] Completed in \033[1m{(time.time()-global_timer)/3600} Hours\033[0m')
    cap.release()
    if args['save'] or args['hide'] is False:
        out_vid.release()
        print(f"[INFO] Outout Video Saved in \033[1m{path_save}\033[0m")
    if args['hide']:
        cv2.destroyAllWindows()
