"""
Model Training Script
Train YOLOv8, YOLOv11, or RT-DETR models on custom datasets.

Capabilities:
1. Train Models: Supports YOLOv8, YOLOv11, RT-DETR (n/s/m/l/x variants)
2. Auto-Validation: Automatically runs validation on the best model after training and saves to `train_eval/`
3. Auto-Download: Fetches pretrained weights if not found locally
4. Path Management: Automatically resolves dataset paths relative to the project structure

Usage Examples:
    python3 scripts/train.py --model yolov8n --epochs 100
    python3 scripts/train.py --model yolo11x --batch-size 16 --workers 8
    python3 scripts/train.py --model rtdetr-l --data ../dataset/custom.yaml
"""

import argparse
import yaml
import os
import sys
import logging
import time
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


def get_dataset_counts(config: dict) -> dict:
    """Count image files in each dataset split."""
    counts = {}
    for split in ['train', 'val', 'test']:
        path_str = config['dataset'].get(split)
        if path_str:
            img_dir = Path(path_str) / "images"
            if img_dir.exists():
                # Count common image files
                count = len([f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']])
                counts[split] = count
            else:
                counts[split] = 0
        else:
            counts[split] = 0
    return counts


def train_generic(config: dict, model_name: str, dataset_counts: dict) -> tuple[str, float, str]:
    """Train a generic model supported by Ultralytics."""
    print(f"\n{'='*60}")
    print(f"Training Model ({model_name})")
    print(f"{'='*60}\n")
    
    start_time = time.time()

    # Determine which class to use (RTDETR has its own class, everything else uses YOLO)
    if "rtdetr" in model_name.lower():
        ModelClass = RTDETR
    else:
        ModelClass = YOLO

    # Load model (locally or from Ultralytics hub)
    pretrained_dir = Path(__file__).parent.parent / "models" / "pretrained"
    os.makedirs(pretrained_dir, exist_ok=True)
    
    pretrained_path = pretrained_dir / f"{model_name}.pt"
    
    if pretrained_path.exists():
        print(f"Loading local model from {pretrained_path}")
        model = ModelClass(str(pretrained_path))
    else:
        print(f"Model not found at {pretrained_path}, downloading...")
        # Ultralytics downloads to current working directory by default
        model = ModelClass(model_name)
        
        # Move downloaded file to pretrained directory
        cwd_download = Path.cwd() / f"{model_name}.pt"
        if cwd_download.exists():
            print(f"Moving downloaded model to {pretrained_path}")
            try:
                os.rename(cwd_download, pretrained_path)
                # Reload from new location to ensure path consistency
                model = ModelClass(str(pretrained_path))
            except Exception as e:
                print(f"Warning: Failed to move model file: {e}")
        else:
            # Maybe it wasn't downloaded to CWD or logic differs
            print(f"Note: Could not locate downloaded file at {cwd_download}")

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
        name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        verbose=False,  # Suppress verbose progress bars
        # Basic augmentation (supported by most YOLO/RTDETR models)
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
    
    # Run explicit validation on best model and save to separate folder
    try:
        save_dir = Path(results.save_dir)
        best_pt = save_dir / "weights" / "best.pt"
        
        print(f"\nRunning final validation for {model_name}...")
        if best_pt.exists():
            val_model = ModelClass(str(best_pt))
            val_model.val(
                split='val',
                project=str(save_dir),
                name='train_eval',
                plots=True
            )
    except Exception as e:
        print(f"Warning: Failed to run explicit validation: {e}")

    # Calculate duration
    end_time = time.time()
    duration = end_time - start_time
    
    # Format summary message
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    
    time_parts = []
    if hours > 0: time_parts.append(f"{hours}hr")
    if minutes > 0: time_parts.append(f"{minutes}min")
    time_parts.append(f"{seconds}s")
    time_str = " ".join(time_parts)
    
    summary_str = (
        f"Model: {model_name}, "
        f"Dataset size: {dataset_counts.get('train', 0)} images (train), "
        f"{dataset_counts.get('val', 0)} (val), "
        f"{dataset_counts.get('test', 0)} (test), "
        f"Epochs: {config['training']['epochs']}, "
        f"Batch Size: {config['training']['batch_size']}, "
        f"Workers: {config['training']['workers']}, "
        f"Training Time: {time_str}"
    )
    
    # Save summary to file
    try:
        summary_file = Path(results.save_dir) / "training_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_str + "\n")
        print(f"Summary saved to {summary_file}")
    except Exception as e:
        print(f"Warning: Failed to save summary file: {e}")

    print(f"\nTraining for {model_name} completed in {time_str}")
    
    return str(results.save_dir), duration, summary_str


def main():
    parser = argparse.ArgumentParser(description="Train BFMC Vision models")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model to train (e.g., yolov8, yolo11, rtdetr, or specific like yolov10n) (default: all)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset YAML file (overrides config)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent.parent / "configs" / "config.yaml"),
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override number of dataloader workers"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.workers:
        config['training']['workers'] = args.workers

    # Create output directories (relative to project root)
    script_dir = Path(__file__).parent.parent
    project_root = Path(__file__).resolve().parent.parent.parent
    results_dir = project_root / config['output']['results_dir']
    models_dir = project_root / config['output']['models_dir']
    
    # Update config with absolute paths
    config['output']['results_dir'] = str(results_dir)
    config['output']['models_dir'] = str(models_dir)

    # Resolve dataset path relative to training directory (if not overridden by absolute path or args)
    if not args.data:
        # If it's a relative path in config, make it relative to training/ directory
        data_yaml_path = Path(config['dataset']['data_yaml'])
        if not data_yaml_path.is_absolute():
            config['dataset']['data_yaml'] = str((script_dir / data_yaml_path).resolve())

    os.makedirs(config['output']['results_dir'], exist_ok=True)
    os.makedirs(config['output']['models_dir'], exist_ok=True)

    # Get dataset counts
    dataset_counts = get_dataset_counts(config)

    # Track results
    results_data = {}

    # Train selected models
    if args.model == "all":
        for model_name in config['models']['list']:
            results_data[model_name] = train_generic(config, model_name, dataset_counts)
    else:
        # Train specific model (can be any name supported by Ultralytics)
        results_data[args.model] = train_generic(config, args.model, dataset_counts)

    # Summary
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    for model, (path, duration, summary) in results_data.items():
        print(summary)
        print(f"Path: {path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
