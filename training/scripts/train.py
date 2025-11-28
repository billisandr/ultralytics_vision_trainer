#!/usr/bin/env python3
"""
BFMC Vision Model Training Script
Train YOLOv8, YOLOv11, or RT-DETR models on BFMC dataset.
"""

import argparse
import yaml
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO, RTDETR

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Suppress ultralytics verbose output
os.environ['YOLO_VERBOSE'] = 'False'


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_yolov8(config: dict, model_size: str = None) -> str:
    """Train YOLOv8 model."""
    model_name = model_size or config['models']['yolov8']['name']
    print(f"\n{'='*60}")
    print(f"Training YOLOv8 ({model_name})")
    print(f"{'='*60}\n")

    # Load pretrained model from models/pretrained/ directory
    pretrained_path = Path(__file__).parent.parent / "models" / "pretrained" / f"{model_name}.pt"
    model = YOLO(str(pretrained_path))

    # Train
    results = model.train(
        data=config['dataset']['data_yaml'],
        epochs=config['training']['epochs'],
        batch=config['training']['batch_size'],
        imgsz=config['training']['imgsz'],
        patience=config['training']['patience'],
        workers=config['training']['workers'],
        device=config['training']['device'],
        amp=config['training']['amp'],
        cache=config['training']['cache'],
        project=config['output']['results_dir'],
        name=f"yolov8_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        verbose=False,  # Suppress verbose progress bars
        # Augmentation
        hsv_h=config['augmentation']['hsv_h'],
        hsv_s=config['augmentation']['hsv_s'],
        hsv_v=config['augmentation']['hsv_v'],
        degrees=config['augmentation']['degrees'],
        translate=config['augmentation']['translate'],
        scale=config['augmentation']['scale'],
        flipud=config['augmentation']['flipud'],
        fliplr=config['augmentation']['fliplr'],
        mosaic=config['augmentation']['mosaic'],
        mixup=config['augmentation']['mixup'],
    )

    return str(results.save_dir)


def train_yolov11(config: dict, model_size: str = None) -> str:
    """Train YOLOv11 model."""
    model_name = model_size or config['models']['yolov11']['name']
    print(f"\n{'='*60}")
    print(f"Training YOLOv11 ({model_name})")
    print(f"{'='*60}\n")

    # Load pretrained model from models/pretrained/ directory
    pretrained_path = Path(__file__).parent.parent / "models" / "pretrained" / f"{model_name}.pt"
    model = YOLO(str(pretrained_path))

    # Train
    results = model.train(
        data=config['dataset']['data_yaml'],
        epochs=config['training']['epochs'],
        batch=config['training']['batch_size'],
        imgsz=config['training']['imgsz'],
        patience=config['training']['patience'],
        workers=config['training']['workers'],
        device=config['training']['device'],
        amp=config['training']['amp'],
        cache=config['training']['cache'],
        project=config['output']['results_dir'],
        name=f"yolov11_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        verbose=False,  # Suppress verbose progress bars
        # Augmentation
        hsv_h=config['augmentation']['hsv_h'],
        hsv_s=config['augmentation']['hsv_s'],
        hsv_v=config['augmentation']['hsv_v'],
        degrees=config['augmentation']['degrees'],
        translate=config['augmentation']['translate'],
        scale=config['augmentation']['scale'],
        flipud=config['augmentation']['flipud'],
        fliplr=config['augmentation']['fliplr'],
        mosaic=config['augmentation']['mosaic'],
        mixup=config['augmentation']['mixup'],
    )

    return str(results.save_dir)


def train_rtdetr(config: dict, model_size: str = None) -> str:
    """Train RT-DETR model."""
    model_name = model_size or config['models']['rtdetr']['name']
    print(f"\n{'='*60}")
    print(f"Training RT-DETR ({model_name})")
    print(f"{'='*60}\n")

    # Load pretrained model from models/pretrained/ directory
    pretrained_path = Path(__file__).parent.parent / "models" / "pretrained" / f"{model_name}.pt"
    model = RTDETR(str(pretrained_path))

    # Train
    results = model.train(
        data=config['dataset']['data_yaml'],
        epochs=config['training']['epochs'],
        batch=config['training']['batch_size'],
        imgsz=config['training']['imgsz'],
        patience=config['training']['patience'],
        workers=config['training']['workers'],
        device=config['training']['device'],
        amp=config['training']['amp'],
        cache=config['training']['cache'],
        project=config['output']['results_dir'],
        name=f"rtdetr_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        verbose=False,  # Suppress verbose progress bars
    )

    return str(results.save_dir)


def main():
    parser = argparse.ArgumentParser(description="Train BFMC Vision models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["yolov8", "yolov11", "rtdetr", "all"],
        default="all",
        help="Model to train (default: all)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--size",
        type=str,
        default=None,
        help="Model size override (e.g., yolov8s, yolo11m, rtdetr-x)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    # Create output directories
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    os.makedirs(config['output']['models_dir'], exist_ok=True)

    # Track results
    results_paths = {}

    # Train selected models
    if args.model in ["yolov8", "all"]:
        results_paths['yolov8'] = train_yolov8(config, args.size if args.model == "yolov8" else None)

    if args.model in ["yolov11", "all"]:
        results_paths['yolov11'] = train_yolov11(config, args.size if args.model == "yolov11" else None)

    if args.model in ["rtdetr", "all"]:
        results_paths['rtdetr'] = train_rtdetr(config, args.size if args.model == "rtdetr" else None)

    # Summary
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    for model, path in results_paths.items():
        print(f"{model}: {path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
