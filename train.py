from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training import Trainer
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
import argparse
import yaml
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--data", type=str, required=True,
                help="path to data.yaml")
ap.add_argument("-n", "--name", type=str,
                help="Checkpoint dir name")
ap.add_argument("-b", "--batch", type=int, default=6,
                help="Training batch size")
ap.add_argument("-e", "--epoch", type=int, default=100,
                help="Training number of epochs")
ap.add_argument("-j", "--worker", type=int, default=2,
                help="Training number of workers")
ap.add_argument("-m", "--model", type=str, required=True,
                help="Model type (eg: yolo_nas_s)")
ap.add_argument("-w", "--weight", type=str, default='coco',
                help="path to pre-trained model weight")

args = vars(ap.parse_args())

if args['name'] is None:
    name = 'train'
else:
    name = args['name']
n = 0
while True:
    if not os.path.exists(os.path.join('runs', f'{name}{n}')):
        if n > 0:
            name = f'{name}{n}'
        os.makedirs(os.path.join('runs', name))
        print(f"[INFO] Checkpoints saved in {os.path.join('runs', name)}")
        break
    else:
        n += 1

trainer = Trainer(experiment_name=name, ckpt_root_dir='runs')
yaml_params = yaml.safe_load(open(args['data'], 'r'))

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': yaml_params['Dir'],
        'images_dir': yaml_params['images']['train'],
        'labels_dir': yaml_params['labels']['train'],
        'classes': yaml_params['names']
    },
    dataloader_params={
        'batch_size': args['batch'],
        'num_workers': args['worker']
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': yaml_params['Dir'],
        'images_dir': yaml_params['images']['val'],
        'labels_dir': yaml_params['labels']['val'],
        'classes': yaml_params['names']
    },
    dataloader_params={
        'batch_size':args['batch'],
        'num_workers': args['worker']
    }
)

test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': yaml_params['Dir'],
        'images_dir': yaml_params['images']['test'],
        'labels_dir': yaml_params['labels']['test'],
        'classes': yaml_params['names']
    },
    dataloader_params={
        'batch_size':args['batch'],
        'num_workers': args['worker']
    }
)

model = models.get(
    args['model'],
    num_classes=len(yaml_params['names']), 
    pretrained_weights=args["weight"]
)

train_params = {
    # ENABLING SILENT MODE
    'silent_mode': False,
    "average_best_models":True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    # ONLY TRAINING FOR 10 EPOCHS FOR THIS EXAMPLE NOTEBOOK
    "max_epochs": args['epoch'],
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        # NOTE: num_classes needs to be defined here
        num_classes=len(yaml_params['names']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            # NOTE: num_classes needs to be defined here
            num_cls=len(yaml_params['names']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50'
}

trainer.train(
    model=model, 
    training_params=train_params, 
    train_loader=train_data, 
    valid_loader=val_data
)


# Evaluating on Test Dataset
best_model = models.get(args['model'],
                        num_classes=len(yaml_params['names']),
                        checkpoint_path=os.path.join('runs', name, 'ckpt_best.pth'))
trainer.test(model=best_model,
            test_loader=test_data,
            test_metrics_list=DetectionMetrics_050(score_thres=0.1, 
                                                   top_k_predictions=300, 
                                                   num_cls=len(yaml_params['names']), 
                                                   normalize_targets=True, 
                                                   post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, 
                                                                                                          nms_top_k=1000, 
                                                                                                          max_predictions=300,                                                                              
                                                                                                          nms_threshold=0.7)
                                                  ))

