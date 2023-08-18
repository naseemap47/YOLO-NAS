from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training import Trainer
from super_gradients.training import models
import argparse
import torch
import time
import yaml
import os


if __name__ == '__main__':

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
    ap.add_argument("-m", "--model", type=str, default='yolo_nas_s',
                choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'],
                help="Model type (eg: yolo_nas_s)")
    ap.add_argument("-w", "--weight", type=str, default='coco',
                    help="path to pre-trained model weight")
    ap.add_argument("-s", "--size", type=int, default=640,
                    help="input image size")
    ap.add_argument("--gpus", action='store_true',
                help="Run on all gpus")
    ap.add_argument("--cpu", action='store_true',
                help="Run on CPU")
    
    
    # train_params
    ap.add_argument("--warmup_mode", type=str, default='linear_epoch_step',
                    help="Warmup Mode")
    ap.add_argument("--warmup_initial_lr", type=float, default=1e-6,
                    help="Warmup Initial LR")
    ap.add_argument("--lr_warmup_epochs", type=int, default=3,
                    help="LR Warmup Epochs")
    ap.add_argument("--initial_lr", type=float, default=5e-4,
                    help="Inital LR")
    ap.add_argument("--lr_mode", type=str, default='cosine',
                    help="LR Mode")
    ap.add_argument("--cosine_final_lr_ratio", type=float, default=0.1,
                    help="Cosine Final LR Ratio")
    ap.add_argument("--optimizer", type=str, default='Adam',
                    help="Optimizer")
    ap.add_argument("--weight_decay", type=float, default=0.0001,
                    help="Weight Decay")
    args = vars(ap.parse_args())

    s_time = time.time()

    if args['name'] is None:
        name = 'train'
    else:
        name = args['name']
    n = 0
    while True:
        if not os.path.exists(os.path.join('runs', f'{name}{n}')):
            name = f'{name}{n}'
            os.makedirs(os.path.join('runs', name))
            print(f"[INFO] Checkpoints saved in \033[1m{os.path.join('runs', name)}\033[0m")
            break
        else:
            n += 1

    # Training on GPU or CPU
    if args['cpu']:
        print('[INFO] Training on \033[1mCPU\033[0m')
        trainer = Trainer(experiment_name=name, ckpt_root_dir='runs', device='cpu')
    elif args['gpus']:
        print(f'[INFO] Training on GPU: \033[1m{torch.cuda.get_device_name()}\033[0m')
        trainer = Trainer(experiment_name=name, ckpt_root_dir='runs', multi_gpu=args['gpus'])
    else:
        print(f'[INFO] Training on GPU: \033[1m{torch.cuda.get_device_name()}\033[0m')
        trainer = Trainer(experiment_name=name, ckpt_root_dir='runs')

    yaml_params = yaml.safe_load(open(args['data'], 'r'))

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': yaml_params['Dir'],
            'images_dir': yaml_params['images']['train'],
            'labels_dir': yaml_params['labels']['train'],
            'classes': yaml_params['names'],
            'input_dim': (args['size'], args['size'])
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
            'classes': yaml_params['names'],
            'input_dim': (args['size'], args['size'])
        },
        dataloader_params={
            'batch_size':args['batch'],
            'num_workers': args['worker']
        }
    )

    if 'test' in (yaml_params['images'].keys() or yaml_params['labels'].keys()):
        test_data = coco_detection_yolo_format_val(
            dataset_params={
                'data_dir': yaml_params['Dir'],
                'images_dir': yaml_params['images']['test'],
                'labels_dir': yaml_params['labels']['test'],
                'classes': yaml_params['names'],
                'input_dim': (args['size'], args['size'])
            },
            dataloader_params={
                'batch_size':args['batch'],
                'num_workers': args['worker']
            }
        )

    if not args["weight"]:
        weights = None
    else:
        weights = args["weights"]

    model = models.get(
        args['model'],
        num_classes=len(yaml_params['names']), 
        pretrained_weights=weights
    )

    train_params = {
        # ENABLING SILENT MODE
        'silent_mode': False,
        "average_best_models":True,
        "warmup_mode": args['warmup_mode'],
        "warmup_initial_lr": args['warmup_initial_lr'],
        "lr_warmup_epochs": args['lr_warmup_epochs'],
        "initial_lr": args['initial_lr'],
        "lr_mode": args['lr_mode'],
        "cosine_final_lr_ratio": args['cosine_final_lr_ratio'],
        "optimizer": args['optimizer'],
        "optimizer_params": {"weight_decay": args['weight_decay']},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": args['epoch'],
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
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

    # Load best model
    best_model = models.get(args['model'],
                            num_classes=len(yaml_params['names']),
                            checkpoint_path=os.path.join('runs', name, 'ckpt_best.pth'))
    
    # Evaluating on Val Dataset
    eval_model = trainer.test(model=best_model,
                    test_loader=val_data,
                    test_metrics_list=DetectionMetrics_050(score_thres=0.1, 
                                                        top_k_predictions=300, 
                                                        num_cls=len(yaml_params['names']), 
                                                        normalize_targets=True, 
                                                        post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01, 
                                                                                                                nms_top_k=1000, 
                                                                                                                max_predictions=300,                                                                              
                                                                                                                nms_threshold=0.7)
                                                        ))
    print('\033[1m [INFO] Validating Model:\033[0m')
    for i in eval_model:
        print(f"{i}: {float(eval_model[i])}")

    # Evaluating on Test Dataset
    if 'test' in (yaml_params['images'].keys() or yaml_params['labels'].keys()):
        test_result = trainer.test(model=best_model,
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
        print('\033[1m [INFO] Test Results:\033[0m')
        for i in test_result:
            print(f"{i}: {float(test_result[i])}")
    print(f'[INFO] Training Completed in \033[1m{(time.time()-s_time)/3600} Hours\033[0m')
    
