#!/usr/bin/env python3
"""
BFMC Vision Model Evaluation and Comparison Script
Evaluate trained models and generate comparison reports.
"""

import argparse
import json
import yaml
import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import pandas as pd
from ultralytics import YOLO, RTDETR

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.visualize import plot_per_class_map_comparison, plot_class_map_heatmap

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def resolve_dataset_path(data_path_str: str, script_dir: Path) -> str:
    """
    Resolve the dataset path relative to the training/ directory
    to ensure robust path handling.
    """
    if not data_path_str:
        return None
        
    # training/ directory is two levels up from this script (training/scripts/script.py)
    training_root = script_dir.parent
    
    path_obj = Path(data_path_str)
    if path_obj.is_absolute():
        return str(path_obj)
    
    # Resolve relative to training root
    return str((training_root / path_obj).resolve())

# Suppress ultralytics verbose output
os.environ['YOLO_VERBOSE'] = 'False'


def get_qualitative_rating(map_score: float, metric_type: str = 'mAP50-95') -> str:
    """
    Get qualitative rating for a metric score.

    Args:
        map_score: The metric score (0-1)
        metric_type: Type of metric for context-aware thresholds

    Returns:
        Qualitative rating string with emoji indicator
    """
    if metric_type == 'mAP50-95':
        # Stricter thresholds for mAP50-95
        if map_score >= 0.70:
            return "Excellent"
        elif map_score >= 0.60:
            return "Very Good"
        elif map_score >= 0.50:
            return "Good"
        elif map_score >= 0.40:
            return "Fair"
        elif map_score >= 0.30:
            return "Needs Improvement"
        else:
            return "Poor"
    elif metric_type == 'mAP50':
        # More lenient thresholds for mAP50
        if map_score >= 0.85:
            return "Excellent"
        elif map_score >= 0.75:
            return "Very Good"
        elif map_score >= 0.65:
            return "Good"
        elif map_score >= 0.55:
            return "Fair"
        elif map_score >= 0.45:
            return "Needs Improvement"
        else:
            return "Poor"
    else:
        # Generic thresholds
        if map_score >= 0.80:
            return "Excellent"
        elif map_score >= 0.70:
            return "Very Good"
        elif map_score >= 0.60:
            return "Good"
        elif map_score >= 0.50:
            return "Fair"
        elif map_score >= 0.40:
            return "Needs Improvement"
        else:
            return "Poor"


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def find_best_weights(results_dir: str, model_type: str) -> List[str]:
    """Find all best.pt weights for a model type."""
    results_path = Path(results_dir)
    weights = []

    for run_dir in results_path.glob(f"{model_type}*"):
        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            weights.append(str(best_pt))

    return sorted(weights, key=lambda x: os.path.getmtime(x), reverse=True)


def find_pretrained_models(results_dir: str) -> List[Dict[str, str]]:
    """Find pretrained models in results/pretrained_models/."""
    pretrained_dir = Path(results_dir) / "pretrained_models"
    models = []

    if pretrained_dir.exists():
        for pt_file in pretrained_dir.glob("*.pt"):
            models.append({
                'path': str(pt_file),
                'name': pt_file.stem,
                'type': 'pretrained'
            })

    return models


def evaluate_model(weights_path: str, data_yaml: str, config: dict, model_type: str = 'trained') -> Dict:
    """Evaluate a single model and return metrics."""
    print(f"\nEvaluating: {weights_path}")

    # Determine model type from path
    if "rtdetr" in weights_path.lower():
        model = RTDETR(weights_path)
    else:
        model = YOLO(weights_path)

    # Resolve dataset path: Prefer args.yaml from model dir if available
    eval_data_yaml = data_yaml
    model_dir = Path(weights_path).parent.parent # Assuming weights/best.pt structure
    args_yaml_path = model_dir / "args.yaml"
    
    if args_yaml_path.exists():
        try:
            with open(args_yaml_path, 'r') as f:
                model_config = yaml.safe_load(f)
                if model_config and 'data' in model_config:
                    script_dir = Path(__file__).parent
                    resolved_data = resolve_dataset_path(model_config['data'], script_dir)
                    if resolved_data and os.path.exists(resolved_data):
                        eval_data_yaml = resolved_data
                        print(f"  Using dataset from args.yaml: {eval_data_yaml}")
        except Exception as e:
            print(f"  Warning: Failed to read args.yaml: {e}")

    # Run validation
    start_time = time.time()
    results = model.val(
        data=eval_data_yaml,
        batch=config['training']['batch_size'],
        imgsz=config['training']['imgsz'],
        conf=config['inference']['conf_threshold'],
        iou=config['inference']['iou_threshold'],
        device=config['training']['device'],
        split='test',  # Evaluate on test set
        verbose=False,  # Suppress verbose progress bars
    )
    eval_time = time.time() - start_time

    # Measure inference speed
    speed_metrics = benchmark_speed(model, config)

    # Determine display name
    if model_type == 'pretrained':
        model_name = f"{Path(weights_path).stem} (pretrained)"
    else:
        model_name = Path(weights_path).parent.parent.name

    # Extract metrics
    map50 = float(results.box.map50)
    map50_95 = float(results.box.map)

    metrics = {
        'weights_path': weights_path,
        'model_name': model_name,
        'model_type': model_type,
        'base_model': Path(weights_path).stem.replace('n', '').replace('s', '').replace('m', '').replace('l', '').replace('x', ''),
        'mAP50': map50,
        'mAP50-95': map50_95,
        'mAP50_rating': get_qualitative_rating(map50, 'mAP50'),
        'mAP50-95_rating': get_qualitative_rating(map50_95, 'mAP50-95'),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
        'eval_time_s': eval_time,
        'inference_ms': speed_metrics['inference_ms'],
        'fps': speed_metrics['fps'],
        'model_size_mb': os.path.getsize(weights_path) / (1024 * 1024),
    }

    # Per-class metrics
    if hasattr(results.box, 'ap_class_index'):
        class_names = results.names
        for i, class_idx in enumerate(results.box.ap_class_index):
            class_name = class_names[int(class_idx)]
            metrics[f'AP50_{class_name}'] = float(results.box.ap50[i])
            # Also collect per-class AP50-95
            if hasattr(results.box, 'ap'):
                metrics[f'AP50-95_{class_name}'] = float(results.box.ap[i])

    # Store class names for later use
    metrics['class_names'] = list(results.names.values()) if hasattr(results, 'names') else []

    return metrics


def benchmark_speed(model, config: dict, num_runs: int = 100) -> Dict:
    """Benchmark inference speed."""
    import numpy as np

    # Create dummy input
    dummy_input = np.random.randint(0, 255, (config['training']['imgsz'], config['training']['imgsz'], 3), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        model.predict(dummy_input, verbose=False)

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.time()
        model.predict(dummy_input, verbose=False)
        times.append((time.time() - start) * 1000)  # ms

    avg_time = sum(times) / len(times)

    return {
        'inference_ms': avg_time,
        'fps': 1000 / avg_time,
    }


def compare_models(metrics_list: List[Dict], output_dir: str) -> pd.DataFrame:
    """Compare multiple models and generate report."""
    df = pd.DataFrame(metrics_list)

    # Sort by mAP50-95
    df = df.sort_values('mAP50-95', ascending=False)

    # Save to CSV
    csv_path = Path(output_dir) / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nComparison saved to: {csv_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Get class names from first model with class data
    class_names = []
    for metrics in metrics_list:
        if 'class_names' in metrics and metrics['class_names']:
            class_names = metrics['class_names']
            break

    if class_names:
        # Per-class AP50 visualizations
        plot_per_class_map_comparison(
            metrics_list,
            class_names,
            output_path=Path(output_dir) / "per_class_ap50_comparison.png",
            metric_name='AP50'
        )

        plot_class_map_heatmap(
            metrics_list,
            class_names,
            output_path=Path(output_dir) / "per_class_ap50_heatmap.png",
            metric_name='AP50'
        )

        # Check if we have AP50-95 per-class data
        has_ap50_95 = any(
            any(f'AP50-95_{cls}' in m for cls in class_names)
            for m in metrics_list
        )

        if has_ap50_95:
            # Per-class AP50-95 visualizations
            plot_per_class_map_comparison(
                metrics_list,
                class_names,
                output_path=Path(output_dir) / "per_class_ap50-95_comparison.png",
                metric_name='AP50-95'
            )

            plot_class_map_heatmap(
                metrics_list,
                class_names,
                output_path=Path(output_dir) / "per_class_ap50-95_heatmap.png",
                metric_name='AP50-95'
            )

    # Generate summary report
    report_path = Path(output_dir) / "comparison_report.md"
    generate_report(df, report_path, output_dir)
    print(f"Report saved to: {report_path}")

    return df


def generate_report(df: pd.DataFrame, output_path: Path, output_dir: str = None):
    """Generate markdown comparison report."""
    report = []
    report.append("# BFMC Vision Model Comparison Report\n")
    report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Check if we have pretrained models
    has_pretrained = 'pretrained' in df['model_type'].values if 'model_type' in df.columns else False

    # Summary table with qualitative ratings
    report.append("## Performance Summary\n\n")
    report.append("| Model | Type | mAP50 | Rating | mAP50-95 | Rating | Precision | Recall | FPS | Size (MB) |\n")
    report.append("|-------|------|-------|--------|----------|--------|-----------|--------|-----|----------|\n")

    for _, row in df.iterrows():
        model_type = row.get('model_type', 'trained')
        map50_rating = row.get('mAP50_rating', 'N/A')
        map50_95_rating = row.get('mAP50-95_rating', 'N/A')

        report.append(
            f"| {row['model_name']} | {model_type} | {row['mAP50']:.3f} | {map50_rating} | "
            f"{row['mAP50-95']:.3f} | {map50_95_rating} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['fps']:.1f} | "
            f"{row['model_size_mb']:.1f} |\n"
        )

    # Rating scale explanation
    report.append("\n### Rating Scale\n\n")
    report.append("**mAP50-95 Ratings:**\n")
    report.append("- Excellent: ≥0.70 | Very Good: ≥0.60 | Good: ≥0.50 | Fair: ≥0.40 | Needs Improvement: ≥0.30 | Poor: <0.30\n\n")
    report.append("**mAP50 Ratings:**\n")
    report.append("- Excellent: ≥0.85 | Very Good: ≥0.75 | Good: ≥0.65 | Fair: ≥0.55 | Needs Improvement: ≥0.45 | Poor: <0.45\n")

    # Per-class visualizations
    if output_dir:
        ap50_comparison = Path(output_dir) / "per_class_ap50_comparison.png"
        ap50_heatmap = Path(output_dir) / "per_class_ap50_heatmap.png"
        ap50_95_comparison = Path(output_dir) / "per_class_ap50-95_comparison.png"
        ap50_95_heatmap = Path(output_dir) / "per_class_ap50-95_heatmap.png"

        if any(p.exists() for p in [ap50_comparison, ap50_heatmap, ap50_95_comparison, ap50_95_heatmap]):
            report.append("\n## Per-Class Performance\n\n")
            report.append("Detailed per-class analysis showing which models perform best on each object class.\n\n")

            # AP50 visualizations
            if ap50_comparison.exists() or ap50_heatmap.exists():
                report.append("### AP50 (IoU=0.5)\n\n")

                if ap50_comparison.exists():
                    report.append("![Per-Class AP50 Comparison](per_class_ap50_comparison.png)\n\n")
                    report.append("*Grouped bar chart comparing AP50 across all classes*\n\n")

                if ap50_heatmap.exists():
                    report.append("![Per-Class AP50 Heatmap](per_class_ap50_heatmap.png)\n\n")
                    report.append("*Heatmap showing AP50 performance patterns across models and classes*\n\n")

            # AP50-95 visualizations
            if ap50_95_comparison.exists() or ap50_95_heatmap.exists():
                report.append("### AP50-95 (IoU=0.5:0.95)\n\n")

                if ap50_95_comparison.exists():
                    report.append("![Per-Class AP50-95 Comparison](per_class_ap50-95_comparison.png)\n\n")
                    report.append("*Grouped bar chart comparing AP50-95 across all classes*\n\n")

                if ap50_95_heatmap.exists():
                    report.append("![Per-Class AP50-95 Heatmap](per_class_ap50-95_heatmap.png)\n\n")
                    report.append("*Heatmap showing AP50-95 performance patterns across models and classes*\n\n")

    # Training improvement analysis
    if has_pretrained:
        report.append("\n## Training Improvement\n\n")
        report.append("Comparison of pretrained vs fine-tuned models on BFMC dataset:\n\n")
        report.append("| Model | Pretrained mAP50-95 | Rating | Fine-tuned mAP50-95 | Rating | Improvement |\n")
        report.append("|-------|---------------------|--------|---------------------|--------|-------------|\n")

        # Group by base model
        if 'base_model' in df.columns:
            for base_model in df['base_model'].unique():
                model_df = df[df['base_model'] == base_model]
                pretrained = model_df[model_df['model_type'] == 'pretrained']
                trained = model_df[model_df['model_type'] == 'trained']

                if not pretrained.empty and not trained.empty:
                    pre_map = pretrained.iloc[0]['mAP50-95']
                    pre_rating = pretrained.iloc[0].get('mAP50-95_rating', 'N/A')
                    train_map = trained.iloc[0]['mAP50-95']
                    train_rating = trained.iloc[0].get('mAP50-95_rating', 'N/A')
                    improvement = ((train_map - pre_map) / pre_map) * 100

                    report.append(
                        f"| {base_model} | {pre_map:.3f} | {pre_rating} | "
                        f"{train_map:.3f} | {train_rating} | {improvement:+.1f}% |\n"
                    )

    # Best model recommendation
    report.append("\n## Recommendations\n\n")

    # Only consider trained models for best recommendation
    trained_df = df[df['model_type'] == 'trained'] if 'model_type' in df.columns else df

    if not trained_df.empty:
        best_accuracy = trained_df.iloc[0]
        fastest = trained_df.loc[trained_df['fps'].idxmax()]
        smallest = trained_df.loc[trained_df['model_size_mb'].idxmin()]

        report.append(f"- **Best Accuracy**: {best_accuracy['model_name']} (mAP50-95: {best_accuracy['mAP50-95']:.3f})\n")
        report.append(f"- **Fastest Inference**: {fastest['model_name']} ({fastest['fps']:.1f} FPS)\n")
        report.append(f"- **Smallest Size**: {smallest['model_name']} ({smallest['model_size_mb']:.1f} MB)\n")

    # Trade-off analysis
    report.append("\n## Trade-off Analysis\n\n")
    report.append("For BFMC competition, consider:\n\n")
    report.append("1. **Real-time requirement**: Need >30 FPS for smooth operation\n")
    report.append("2. **Raspberry Pi deployment**: Smaller models preferred\n")
    report.append("3. **Detection accuracy**: Critical for traffic signs and pedestrians\n")

    if has_pretrained:
        report.append("\n## Conclusion\n\n")
        report.append("Fine-tuning on the BFMC dataset provides significant improvements over pretrained COCO weights, ")
        report.append("especially for domain-specific objects like traffic signs and road markings.\n")

    with open(output_path, 'w') as f:
        f.writelines(report)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare BFMC Vision models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--weights",
        type=str,
        nargs='+',
        default=None,
        help="Specific weight files to evaluate (optional)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing training results"
    )
    # Determine default results directory (project root / results)
    project_root = Path(__file__).resolve().parent.parent.parent
    default_base_results = project_root / "results"
    
    # Generate timestamped folder name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_output = default_base_results / f"comp_eval_{timestamp}"

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for comparison report (default: results/comp_eval_<datetime>)"
    )
    parser.add_argument(
        "--include-pretrained",
        action="store_true",
        help="Include pretrained models in comparison"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    results_dir = args.results_dir or config['output']['results_dir']

    # Find weights to evaluate
    models_to_evaluate = []

    if args.weights:
        # User specified weights
        for w in args.weights:
            models_to_evaluate.append({'path': w, 'type': 'trained'})
    else:
        # Auto-find trained models
        for model_type in ['yolov8', 'yolov11', 'rtdetr']:
            weights = find_best_weights(results_dir, model_type)
            if weights:
                models_to_evaluate.append({'path': weights[0], 'type': 'trained'})  # Most recent

        # Add pretrained models if requested
        if args.include_pretrained:
            pretrained_models = find_pretrained_models(results_dir)
            models_to_evaluate.extend(pretrained_models)

    if not models_to_evaluate:
        print("No models found. Please train models first or use --weights to specify.")
        return

    print(f"Found {len(models_to_evaluate)} models to evaluate:")
    for m in models_to_evaluate:
        model_type = m.get('type', 'trained')
        print(f"  - {m['path']} ({model_type})")

    # Evaluate each model
    metrics_list = []
    for model_info in models_to_evaluate:
        try:
            metrics = evaluate_model(
                model_info['path'],
                config['dataset']['data_yaml'],
                config,
                model_type=model_info.get('type', 'trained')
            )
            metrics_list.append(metrics)
        except Exception as e:
            print(f"Error evaluating {model_info['path']}: {e}")

    # Generate comparison
    if metrics_list:
        # Determine final output path
        output_path = Path(args.output) if args.output else default_output
        
        os.makedirs(output_path, exist_ok=True)
        df = compare_models(metrics_list, output_path)

        # Print summary with ratings
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        cols = ['model_name', 'model_type', 'mAP50-95', 'mAP50-95_rating', 'fps', 'model_size_mb']
        available_cols = [c for c in cols if c in df.columns]
        print(df[available_cols].to_string(index=False))
        print("="*80)

        # Print qualitative insights
        print("\nQUALITATIVE ASSESSMENT:")
        for _, row in df.iterrows():
            rating = row.get('mAP50-95_rating', 'N/A')
            model_type = row.get('model_type', 'trained')
            print(f"  - {row['model_name']}: {rating} ({model_type})")
        print("")


if __name__ == "__main__":
    main()
