from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormatDetectionDataset
from super_gradients.training.transforms.transforms import DetectionMosaic, DetectionRandomAffine, DetectionHSV, \
    DetectionHorizontalFlip, DetectionPaddedRescale, DetectionStandardize, DetectionTargetsFormatTransform
from super_gradients.training.datasets.datasets_utils import worker_init_reset_seed
from super_gradients.training.utils.detection_utils import CrowdDetectionCollateFN
from super_gradients.training.pre_launch_callbacks import modify_params_for_qat
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training import dataloaders
from super_gradients.training import Trainer
from super_gradients.training import models
import argparse
import torch
import time
import yaml
import json
import os



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--data", type=str, required=True,
                    help="path to data.yaml")
    ap.add_argument("-b", "--batch", type=int, default=6,
                    help="Training batch size")
    ap.add_argument("-e", "--epoch", type=int, default=100,
                    help="Training number of epochs")
    ap.add_argument("-j", "--worker", type=int, default=2,
                    help="Training number of workers")
    ap.add_argument("-m", "--model", type=str, default='yolo_nas_s',
                choices=['yolo_nas_s', 'yolo_nas_m', 'yolo_nas_l'],
                help="Model type (eg: yolo_nas_s)")
    ap.add_argument("-w", "--weight", type=str, required=True,
                    help="path to pre-trained model weight [ckpt_best.pth]")
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
    ap.add_argument("--optimizer", type=str, default='AdamW',
                    help="Optimizer")
    ap.add_argument("--weight_decay", type=float, default=0.0001,
                    help="Weight Decay")
    args = vars(ap.parse_args())

    # Quantization Aware Training INFO
    print("\x1b[6;37;41m [INFO] Quantization Aware Training \x1b[0m")
    print("\x1b[1;37;41m [WARNING]: Quantization Aware Training Requires a Large Amount of System RAM \x1b[0m")

    # Start Time
    s_time = time.time()

    # Load Path Params
    yaml_params = yaml.safe_load(open(args['data'], 'r'))
    with open(os.path.join(yaml_params['Dir'], yaml_params['labels']['train'])) as f:
        no_class = len(json.load(f)['categories'])
        f.close()
    print(f"\033[1m[INFO] Number of Classes: {no_class}\033[0m")

    # Training on GPU or CPU    
    _, name = args['weight'].split('/')[-3:-1]
    if args['cpu']:
        print('[INFO] Training on \033[1mCPU\033[0m')
        trainer = Trainer(experiment_name=name, ckpt_root_dir='qat', device='cpu')
    elif args['gpus']:
        print(f'[INFO] Training on GPU: \033[1m{torch.cuda.get_device_name()}\033[0m')
        trainer = Trainer(experiment_name=name, ckpt_root_dir='qat', multi_gpu=args['gpus'])
    else:
        print(f'[INFO] Training on GPU: \033[1m{torch.cuda.get_device_name()}\033[0m')
        trainer = Trainer(experiment_name=name, ckpt_root_dir='qat')

    # Load best model
    best_model = models.get(args['model'],
                            num_classes=no_class,
                            checkpoint_path=args['weight'])
    
    # Reain Dataset
    trainset = COCOFormatDetectionDataset(data_dir=yaml_params['Dir'],
                                      images_dir=yaml_params['images']['train'],
                                      json_annotation_file=yaml_params['labels']['train'],
                                      input_dim=(args['size'], args['size']),
                                      ignore_empty_annotations=False,
                                      transforms=[
                                          DetectionMosaic(prob=1., input_dim=(args['size'], args['size'])),
                                          DetectionRandomAffine(degrees=0., scales=(0.5, 1.5), shear=0.,
                                                                target_size=(args['size'], args['size']),
                                                                filter_box_candidates=False, border_value=128),
                                          DetectionHSV(prob=1., hgain=5, vgain=30, sgain=30),
                                          DetectionHorizontalFlip(prob=0.5),
                                          DetectionPaddedRescale(input_dim=(args['size'], args['size']), max_targets=300),
                                          DetectionStandardize(max_value=255),
                                          DetectionTargetsFormatTransform(max_targets=300, input_dim=(args['size'], args['size']),
                                                                          output_format="LABEL_CXCYWH")
                                      ])
    train_dataloader_params = {
                                "shuffle": True,
                                "batch_size": args['batch'],
                                "drop_last": False,
                                "pin_memory": True,
                                "collate_fn": CrowdDetectionCollateFN(),
                                "worker_init_fn": worker_init_reset_seed,
                                "min_samples": 512
                                }
    # Valid Data
    valset = COCOFormatDetectionDataset(data_dir=yaml_params['Dir'],
                                    images_dir=yaml_params['images']['val'],
                                    json_annotation_file=yaml_params['labels']['val'],
                                    input_dim=(args['size'], args['size']),
                                    ignore_empty_annotations=False,
                                    transforms=[
                                        DetectionPaddedRescale(input_dim=(args['size'], args['size']), max_targets=300),
                                        DetectionStandardize(max_value=255),
                                        DetectionTargetsFormatTransform(max_targets=300, input_dim=(args['size'], args['size']),
                                                                        output_format="LABEL_CXCYWH")
                                    ])
    val_dataloader_params = {
                            "shuffle": False,
                            "batch_size": int(args['batch']*2),
                            "num_workers": args['worker'],
                            "drop_last": False,
                            "pin_memory": True,
                            "collate_fn": CrowdDetectionCollateFN(),
                            "worker_init_fn": worker_init_reset_seed
                            }

    train_params = {
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
            num_classes=no_class,
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=no_class,
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
    
    # Quantization Aware Training
    print("\x1b[1;37;41m [INFO]: Launching Quantization Aware Training \x1b[0m")
    train_params, trainset, valset, train_dataloader_params, val_dataloader_params = modify_params_for_qat(
        train_params, trainset, valset, train_dataloader_params, val_dataloader_params
    )

    # Print Training Params
    print('[INFO] Training Params:\n', train_params)
    
    train_loader = dataloaders.get(dataset=trainset,
                                dataloader_params=train_dataloader_params)
    valid_loader = dataloaders.get(dataset=valset,
                                dataloader_params=val_dataloader_params)
    
    # Quantization Aware Training
    trainer.qat(
        model=best_model, 
        training_params=train_params, 
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        calib_loader=train_loader
    )
    print(f'[INFO] Training Completed in \033[1m{(time.time()-s_time)/3600} Hours\033[0m')
    print("\x1b[1;37;42m [SUCCESS]: Quantization Aware Training Completed \x1b[0m")
