#!/usr/bin/env python3
"""
Visualization utilities for BFMC Vision training results.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Force matplotlib to use non-interactive backend (fixes Qt/display issues)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2


def plot_training_curves(results_dir: str, output_path: Optional[str] = None):
    """
    Plot training curves from results.csv.

    Args:
        results_dir: Path to training results directory
        output_path: Optional path to save the plot
    """
    import pandas as pd

    results_path = Path(results_dir) / "results.csv"
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Training Curves - {Path(results_dir).name}", fontsize=14)

    # Loss curves
    if 'train/box_loss' in df.columns:
        axes[0, 0].plot(df['epoch'], df['train/box_loss'], label='Train')
        if 'val/box_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['val/box_loss'], label='Val')
        axes[0, 0].set_title('Box Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()

    if 'train/cls_loss' in df.columns:
        axes[0, 1].plot(df['epoch'], df['train/cls_loss'], label='Train')
        if 'val/cls_loss' in df.columns:
            axes[0, 1].plot(df['epoch'], df['val/cls_loss'], label='Val')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()

    if 'train/dfl_loss' in df.columns:
        axes[0, 2].plot(df['epoch'], df['train/dfl_loss'], label='Train')
        if 'val/dfl_loss' in df.columns:
            axes[0, 2].plot(df['epoch'], df['val/dfl_loss'], label='Val')
        axes[0, 2].set_title('DFL Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].legend()

    # Metrics
    if 'metrics/precision(B)' in df.columns:
        axes[1, 0].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
        if 'metrics/recall(B)' in df.columns:
            axes[1, 0].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()

    if 'metrics/mAP50(B)' in df.columns:
        axes[1, 1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
        if 'metrics/mAP50-95(B)' in df.columns:
            axes[1, 1].plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
        axes[1, 1].set_title('mAP')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()

    # Learning rate
    if 'lr/pg0' in df.columns:
        axes[1, 2].plot(df['epoch'], df['lr/pg0'])
        axes[1, 2].set_title('Learning Rate')
        axes[1, 2].set_xlabel('Epoch')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: Optional[str] = None,
    normalize: bool = True
):
    """
    Plot confusion matrix.

    Args:
        confusion_matrix: NxN confusion matrix
        class_names: List of class names
        output_path: Optional path to save the plot
        normalize: Normalize by row (true class)
    """
    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
        fmt = '.2f'
    else:
        cm = confusion_matrix
        fmt = 'd'

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_class_distribution(
    class_counts: Dict[str, int],
    output_path: Optional[str] = None,
    title: str = "Class Distribution"
):
    """
    Plot class distribution as bar chart.

    Args:
        class_counts: Dictionary mapping class names to counts
        output_path: Optional path to save the plot
        title: Plot title
    """
    # Sort by count
    sorted_items = sorted(class_counts.items(), key=lambda x: -x[1])
    names = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(names)), counts, color='steelblue')
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)

    # Add value labels
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha='center',
            va='bottom',
            fontsize=8
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved class distribution to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_model_comparison(
    metrics_df,
    output_path: Optional[str] = None
):
    """
    Plot model comparison charts.

    Args:
        metrics_df: DataFrame with model metrics
        output_path: Optional path to save the plot
    """
    import pandas as pd

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Model Comparison', fontsize=14)

    models = metrics_df['model_name'].tolist()
    x = range(len(models))

    # mAP comparison
    if 'mAP50-95' in metrics_df.columns:
        bars = axes[0].bar(x, metrics_df['mAP50-95'], color='steelblue')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].set_ylabel('mAP50-95')
        axes[0].set_title('Accuracy (mAP50-95)')
        for bar, val in zip(bars, metrics_df['mAP50-95']):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # Speed comparison
    if 'fps' in metrics_df.columns:
        bars = axes[1].bar(x, metrics_df['fps'], color='forestgreen')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].set_ylabel('FPS')
        axes[1].set_title('Inference Speed (FPS)')
        for bar, val in zip(bars, metrics_df['fps']):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    # Size comparison
    if 'model_size_mb' in metrics_df.columns:
        bars = axes[2].bar(x, metrics_df['model_size_mb'], color='coral')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(models, rotation=45, ha='right')
        axes[2].set_ylabel('Size (MB)')
        axes[2].set_title('Model Size')
        for bar, val in zip(bars, metrics_df['model_size_mb']):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved model comparison to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_predictions(
    image_path: str,
    predictions: List[Dict],
    class_names: List[str],
    output_path: Optional[str] = None,
    conf_threshold: float = 0.25
):
    """
    Visualize predictions on an image.

    Args:
        image_path: Path to input image
        predictions: List of predictions with 'box', 'class', 'conf' keys
        class_names: List of class names
        output_path: Optional path to save the result
        conf_threshold: Confidence threshold for display
    """
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Generate colors for each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)

    # Draw predictions
    for pred in predictions:
        if pred['conf'] < conf_threshold:
            continue

        box = pred['box']  # [x1, y1, x2, y2]
        cls = pred['class']
        conf = pred['conf']

        color = tuple(map(int, colors[cls]))
        label = f"{class_names[cls]} {conf:.2f}"

        # Draw box
        cv2.rectangle(
            img,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color,
            2
        )

        # Draw label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            img,
            (int(box[0]), int(box[1]) - h - 4),
            (int(box[0]) + w, int(box[1])),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            img,
            label,
            (int(box[0]), int(box[1]) - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predictions: {Path(image_path).name}")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_per_class_map_comparison(
    metrics_list: List[Dict],
    class_names: List[str],
    output_path: Optional[str] = None,
    metric_name: str = 'AP50'
):
    """
    Plot per-class mAP comparison across multiple models.

    Args:
        metrics_list: List of metric dictionaries from evaluate_model()
        class_names: List of all class names
        output_path: Optional path to save the plot
        metric_name: Metric to plot (AP50 or AP50-95)
    """
    import pandas as pd

    # Prepare data for plotting
    data = []
    for metrics in metrics_list:
        model_name = metrics['model_name']
        for class_name in class_names:
            metric_key = f'{metric_name}_{class_name}'
            if metric_key in metrics:
                data.append({
                    'Model': model_name,
                    'Class': class_name,
                    'mAP': metrics[metric_key]
                })

    if not data:
        print(f"No per-class metrics found for {metric_name}")
        return

    df = pd.DataFrame(data)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 8))

    # Get unique models and classes
    models = df['Model'].unique()
    classes = df['Class'].unique()

    # Set up positions for grouped bars
    x = np.arange(len(classes))
    width = 0.8 / len(models)

    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    # Plot bars for each model
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        values = [model_data[model_data['Class'] == cls]['mAP'].values[0]
                  if len(model_data[model_data['Class'] == cls]) > 0 else 0
                  for cls in classes]

        bars = ax.bar(x + i * width, values, width, label=model, color=colors[i], alpha=0.8)

        # Add value labels on bars (only if > 0)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=0)

    # Formatting
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric_name}', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class {metric_name} Comparison Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved per-class comparison to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_class_map_heatmap(
    metrics_list: List[Dict],
    class_names: List[str],
    output_path: Optional[str] = None,
    metric_name: str = 'AP50'
):
    """
    Plot per-class mAP as a heatmap across models.

    Args:
        metrics_list: List of metric dictionaries from evaluate_model()
        class_names: List of all class names
        output_path: Optional path to save the plot
        metric_name: Metric to plot (AP50 or AP50-95)
    """
    import pandas as pd

    # Build matrix: rows=models, cols=classes
    models = [m['model_name'] for m in metrics_list]
    matrix = []

    for metrics in metrics_list:
        row = []
        for class_name in class_names:
            metric_key = f'{metric_name}_{class_name}'
            row.append(metrics.get(metric_key, 0.0))
        matrix.append(row)

    matrix = np.array(matrix)

    # Create heatmap
    plt.figure(figsize=(14, len(models) * 0.6 + 2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        xticklabels=class_names,
        yticklabels=models,
        cbar_kws={'label': f'{metric_name}'},
        vmin=0,
        vmax=1.0,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(f'Per-Class {metric_name} Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved per-class heatmap to {output_path}")
    else:
        plt.show()

    plt.close()


def create_results_dashboard(results_dir: str, output_dir: str):
    """
    Create a complete results dashboard with all visualizations.

    Args:
        results_dir: Path to training results directory
        output_dir: Directory to save dashboard images
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot training curves
    plot_training_curves(
        results_dir,
        os.path.join(output_dir, "training_curves.png")
    )

    # Check for confusion matrix
    cm_path = Path(results_dir) / "confusion_matrix.png"
    if cm_path.exists():
        print(f"Confusion matrix already exists: {cm_path}")

    print(f"\nDashboard created in: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--results", type=str, required=True, help="Results directory")
    parser.add_argument("--output", type=str, default="dashboard", help="Output directory")

    args = parser.parse_args()

    create_results_dashboard(args.results, args.output)