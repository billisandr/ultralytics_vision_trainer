#!/usr/bin/env python3
"""
Post-Training Evaluation Script
Evaluates a trained model checkpoint on Test and Validation sets,
and generates loss/performance curves from training history.

Usage:
    python3 evaluate_checkpoint.py --model_dir path/to/training/result/folder
"""

import argparse
import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO, RTDETR
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resolve_dataset_path(data_path_str, script_dir):
    """
    Resolve the dataset path relative to the training/ directory
    to ensure robust path handling.
    """
    # training/ directory is two levels up from this script (training/scripts/script.py)
    training_root = script_dir.parent
    
    path_obj = Path(data_path_str)
    if path_obj.is_absolute():
        return str(path_obj)
    
    # Resolve relative to training root
    return str((training_root / path_obj).resolve())

def plot_training_history(results_csv_path, output_dir):
    """
    Reads results.csv and plots training vs validation loss curves and metrics.
    """
    if not os.path.exists(results_csv_path):
        logging.warning(f"results.csv not found at {results_csv_path}, skipping history plots.")
        return

    try:
        # Ultralytics results.csv usually has columns with leading spaces
        df = pd.read_csv(results_csv_path)
        df.columns = [c.strip() for c in df.columns]
        
        # Create output directory for plots
        plots_dir = Path(output_dir) / "history_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        epochs = df['epoch']
        
        # 1. Loss Curves
        plt.figure(figsize=(12, 8))
        
        # Box Loss
        if 'train/box_loss' in df.columns and 'val/box_loss' in df.columns:
            plt.subplot(2, 2, 1)
            plt.plot(epochs, df['train/box_loss'], label='Train Box Loss')
            plt.plot(epochs, df['val/box_loss'], label='Val Box Loss')
            plt.title('Box Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # Class Loss
        if 'train/cls_loss' in df.columns and 'val/cls_loss' in df.columns:
            plt.subplot(2, 2, 2)
            plt.plot(epochs, df['train/cls_loss'], label='Train Cls Loss')
            plt.plot(epochs, df['val/cls_loss'], label='Val Cls Loss')
            plt.title('Class Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)

        # DFL Loss (Distribution Focal Loss)
        if 'train/dfl_loss' in df.columns and 'val/dfl_loss' in df.columns:
            plt.subplot(2, 2, 3)
            plt.plot(epochs, df['train/dfl_loss'], label='Train DFL Loss')
            plt.plot(epochs, df['val/dfl_loss'], label='Val DFL Loss')
            plt.title('DFL Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(plots_dir / "loss_curves.png")
        plt.close()
        
        # 2. Metrics Curves
        plt.figure(figsize=(12, 6))
        
        if 'metrics/mAP50(B)' in df.columns:
            plt.subplot(1, 2, 1)
            plt.plot(epochs, df['metrics/mAP50(B)'], label='mAP50')
            if 'metrics/mAP50-95(B)' in df.columns:
                 plt.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP50-95')
            plt.title('mAP Scores')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
        if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
            plt.subplot(1, 2, 2)
            plt.plot(epochs, df['metrics/precision(B)'], label='Precision')
            plt.plot(epochs, df['metrics/recall(B)'], label='Recall')
            plt.title('Precision & Recall')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / "metrics_curves.png")
        plt.close()
        
        logging.info(f"History plots saved to {plots_dir}")
        
    except Exception as e:
        logging.error(f"Error plotting training history: {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model checkpoint")
    parser.add_argument("--model_dir", required=True, type=str, 
                        help="Path to the training run directory (containing args.yaml and weights/)")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        logging.error(f"Directory not found: {model_dir}")
        return

    # 1. Locate Resources
    args_yaml_path = model_dir / "args.yaml"
    weights_path = model_dir / "weights" / "best.pt"
    
    # Fallback for weights location
    if not weights_path.exists():
        weights_path = model_dir / "best.pt"
    
    if not args_yaml_path.exists():
        logging.error(f"args.yaml not found in {model_dir}")
        return
        
    if not weights_path.exists():
        logging.error(f"Weights (best.pt) not found in {model_dir}")
        return

    # 2. Parse Config
    with open(args_yaml_path, 'r') as f:
        train_config = yaml.safe_load(f)
        
    dataset_path_raw = train_config.get('data')
    if not dataset_path_raw:
        logging.error("Could not find 'data' entry in args.yaml")
        return
        
    # Resolve dataset path
    script_dir = Path(__file__).parent
    dataset_path = resolve_dataset_path(dataset_path_raw, script_dir)
    logging.info(f"Resolved dataset path: {dataset_path}")

    # 3. Load Model
    logging.info(f"Loading model from {weights_path}")
    try:
        if "rtdetr" in str(weights_path).lower():
            model = RTDETR(str(weights_path))
        else:
            model = YOLO(str(weights_path))
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # 4. Generate History Plots
    logging.info("Generating training history plots...")
    plot_training_history(model_dir / "results.csv", model_dir)

    # 5. Run Evaluation
    output_dir = model_dir / "post_eval"
    
    # Test Set
    logging.info("Running evaluation on TEST set...")
    model.val(
        data=dataset_path, 
        split='test', 
        project=str(output_dir), 
        name='test',
        plots=True
    )
    
    # Validation Set
    logging.info("Running evaluation on VAL set...")
    # Note: Training often ends with val evaluation, but running it here ensures we get the plots
    # using the exact same settings as our independent test run.
    model.val(
        data=dataset_path, 
        split='val', 
        project=str(output_dir), 
        name='val',
        plots=True
    )
    
    logging.info(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
