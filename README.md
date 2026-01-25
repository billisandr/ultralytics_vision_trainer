# BFMC Vision Training & Evaluation

Bare-bones test code for a training pipeline on object detection models using a BFMC (Bosch Future Mobility Challenge) dataset which features autonomous navigation challenges in smart cities.

> Developed at the [SenseLAB](http://senselab.tuc.gr/) of the [Technical University of Crete](https://www.tuc.gr/el/archi)

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
---

## About BFMC

The [Bosch Future Mobility Challenge](https://boschfuturemobility.com/) is an autonomous driving competition focused on smart city navigation. This project provides training tools for computer vision models to detect objects in the BFMC environment.

---

## Dataset Attribution

This project uses the **BFMC Dataset v13** provided by **Team DriverIES** on Roboflow:

- **Source**: [BFMC Dataset on Roboflow](https://universe.roboflow.com/team-driverles/bfmc-6btkg/dataset/13)
- **License**: CC BY 4.0
- **Credit**: Team DriverIES

---

## AI-Assisted Development

**LLM Usage**: This project was developed with assistance from Claude (Anthropic). While efforts have been made to ensure code quality and correctness:

- **Use with Caution**: Review and test all code thoroughly before deployment
- **No Warranty**: Code is provided "as-is" without guarantees
- **Verify Results**: Always validate model performance and system behavior
- **Safety Critical**: Extra caution required for autonomous driving applications

---

## Overview

Training pipeline for YOLOv8, YOLOv11, and RT-DETR models on the BFMC autonomous driving dataset.

- **Dataset**: 8,509 labeled images with 14 object classes
- **Models**: YOLOv8, YOLOv11, RT-DETR
- **Classes**: Cars, Pedestrians, Traffic Signs (10 types), Stop Lines, Parking Spots, Road Stands

### Supported Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **YOLOv8** | Fast, proven architecture | General purpose, embedded |
| **YOLOv11** | Latest YOLO improvements | Best accuracy/speed trade-off |
| **RT-DETR** | Transformer-based detector | Strong on small objects |

### Example Detection Results

![YOLOv11 detection results on BFMC test images](assets/training/predictions_example.jpg)

---

## Project Structure

```txt
bfmc_vision_simple/
├── training/                    # Model training and evaluation
│   ├── scripts/                 # Train, evaluate, inference scripts
│   │   ├── train.py             # Training script
│   │   ├── evaluate.py          # Model evaluation & comparison
│   │   ├── inference.py         # Inference & export
│   │   ├── train_all_models.sh  # Sequential training
│   │   └── check_split_quality.py # Dataset validation
│   ├── configs/
│   │   └── config.yaml          # Training configuration
│   ├── utils/                   # Dataset analysis, visualization tools
│   │   ├── data.py              # Dataset analysis
│   │   ├── visualize.py         # Per-class plots, heatmaps
│   │   └── video.py             # Video processing
│   ├── results/                 # Training outputs and metrics
│   ├── models/                  # Exported models
│   └── runs/                    # Ultralytics outputs
│
├── dataset/                     # BFMC vision dataset (YOLO format)
│   ├── train/                   # 7,752 training images
│   ├── valid/                   # 437 validation images
│   ├── test/                    # 320 test images
│   └── data.yaml                # Dataset configuration
│
└── assets/                      # Training visualizations
    ├── training/                # Training result images
    └── comparison/              # Model comparison plots
```

---

## Training

### Train All Models

```bash
cd training

# Train all three models sequentially
python3 scripts/train.py --model all

# Or use the shell script
bash scripts/train_all_models.sh
```

### Train Specific Models

```bash
# Train individual models
python3 scripts/train.py --model yolov8
python3 scripts/train.py --model yolov11
python3 scripts/train.py --model rtdetr
```

### Custom Training

```bash
# Different model size
python3 scripts/train.py --model yolov8 --size yolov8s

# Adjust epochs and batch size
python3 scripts/train.py --model yolov11 --epochs 150 --batch-size 32
```

### Training Time (100 epochs)

- **RTX 4090**: ~30-45 minutes
- **RTX 3090**: ~45-60 minutes
- **RTX 3060**: ~1-2 hours

### Expected Performance

```txt
Model: YOLOv8n
├─ mAP50: 0.85 (85% accuracy)
├─ mAP50-95: 0.60
├─ Inference: ~800 FPS (RTX 4090)
└─ Size: 6 MB
```

---

## Evaluation

### Validation vs. Test Metrics

It is important to distinguish between the metrics generated during training and those from the evaluation script:

- **Validation Set (`val`)**: Used by `train.py` during training.
  - The graphs saved in your training results folder (e.g., `results.png`, `confusion_matrix.png`, `BoxPR_curve.png`) reflect performance on the **Validation Set**.
  - These metrics are used to track progress and select the best model checkpoint.

- **Test Set (`test`)**: Used by `evaluate_checkpoint.py` (and `evaluate.py`).
  - The `evaluate_checkpoint.py` script explicitly runs evaluation on the **Test Set** (unseen data).
  - The outputs in the `post_eval/test` folder reflect the model's performance on this held-out dataset, providing the most accurate estimate of real-world performance.

### Compare Models

```bash
cd training

# Compare all trained models
python3 scripts/evaluate.py

# Include pretrained baselines to see improvement
python3 scripts/evaluate.py --include-pretrained

# Evaluate specific weights
python3 scripts/evaluate.py --weights results/yolov8_*/weights/best.pt results/yolov11_*/weights/best.pt
```

### Generated Outputs

- `results/model_comparison.csv` - Complete metrics table
- `results/comparison_report.md` - Detailed report with ratings
- `results/per_class_ap50_comparison.png` - AP50 bar chart
- `results/per_class_ap50_heatmap.png` - AP50 heatmap
- `results/per_class_ap50-95_comparison.png` - AP50-95 bar chart
- `results/per_class_ap50-95_heatmap.png` - AP50-95 heatmap

### Qualitative Ratings

Models receive automatic quality ratings:

- **Excellent** (mAP50-95 ≥0.70): Production-ready
- **Very Good** (≥0.60): High-quality detection
- **Good** (≥0.50): Acceptable performance
- **Fair** (≥0.40): Baseline performance
- **Needs Improvement** (<0.40): Requires more training

Example output:

```txt
Training complete for YOLOv8 (yolov8n)
  mAP50:    0.850 - Excellent
  mAP50-95: 0.620 - Very Good
```

---

## Inference

The `inference.py` script allows you to detecting objects in various inputs and exporting models for deployment.

**Capabilities:**

1. **Run Inference**: Detect objects in images, videos, or a live camera stream.
2. **Export Models**: Convert `.pt` models to ONNX, TensorRT, or TFLite for devices like Raspberry Pi.
3. **Visualize**: Display results with bounding boxes or save them to disk.
4. **Control Speed**: Adjust playback speed for video inspection (`--delay`).

### Run on Test Images

```bash
cd training
python3 scripts/inference.py --weights results/yolov8_*/weights/best.pt
```

### Run on Camera

```bash
python3 scripts/inference.py --weights results/yolov8_*/weights/best.pt --camera 0
```

### Run on Video

```bash
python3 scripts/inference.py --weights results/yolov8_*/weights/best.pt --source video.mp4 --save
```

### Slow Down Playback

```bash
# Add a 50ms delay between frames
python3 scripts/inference.py --weights results/yolov8_*/weights/best.pt --source video.mp4 --delay 50
```

---

## Model Export

### Export to ONNX (Raspberry Pi)

```bash
cd training
python3 scripts/inference.py --weights results/yolov8_*/weights/best.pt --export onnx
```

### Export to TensorRT (Jetson)

```bash
python3 scripts/inference.py --weights results/yolov8_*/weights/best.pt --export engine
```

---

## Dataset

### BFMC Vision Dataset

- **Total Images**: 8,509
- **Classes**: 14
- **Format**: YOLO (txt labels)
- **Resolution**: 640x640
- **Source**: Roboflow BFMC v13

### Class Distribution

| Class | Count | Class | Count |
|-------|-------|-------|-------|
| car | 2,450 | stop-sign | 156 |
| pedestrian | 892 | priority-sign | 289 |
| parking-spot | 1,234 | crosswalk-sign | 445 |
| stop-line | 678 | highway-entry-sign | 123 |

**Split**:

- Training: 7,752 images (91%)
- Validation: 437 images (5%)
- Test: 320 images (4%)

---

## Utilities

### Dataset Analysis

Analyze and validate your dataset:

```python
from utils.data import analyze_dataset, get_class_distribution, validate_labels

# Get dataset statistics
stats = analyze_dataset("../dataset/data.yaml")
print(f"Total images: {stats['total_images']}")
print(f"Classes: {stats['num_classes']}")

# Check class distribution
dist = get_class_distribution("../dataset/data.yaml", split='train')
for cls, count in dist.items():
    print(f"{cls}: {count}")

# Validate labels for issues
results = validate_labels("../dataset/data.yaml")
print(f"Valid: {results['valid_labels']}/{results['total_labels']}")
```

Command-line usage:

```bash
cd training
python3 utils/data.py --data ../dataset/data.yaml --validate --quality
```

### Visualization

Plot training results and model comparisons:

```python
from utils.visualize import plot_training_curves, plot_class_distribution, plot_model_comparison

# Plot training curves from results
plot_training_curves("results/yolov8_best", output_path="curves.png")

# Visualize class distribution
from utils.data import get_class_distribution
dist = get_class_distribution("../dataset/data.yaml")
plot_class_distribution(dist, output_path="distribution.png")

# Compare models (after evaluation)
import pandas as pd
df = pd.read_csv("results/model_comparison.csv")
plot_model_comparison(df, output_path="comparison.png")
```

---

## Configuration

Edit `training/configs/config.yaml` to adjust:

- **Training parameters**: epochs, batch size, image size
- **Model selection**: model variants (n/s/m/l/x)
- **Augmentation**: data augmentation settings
- **Inference**: confidence/IoU thresholds

### RTX 4090 Optimized Settings

The default configuration is optimized for RTX 4090:

- `batch_size: 16` - Can increase to 32 for smaller models
- `imgsz: 640` - Standard BFMC resolution
- `amp: true` - Mixed precision for faster training
- `cache: true` - Cache images in RAM

---

## Training Visualizations

![Training and validation metrics over 100 epochs](assets/training/training_curves.png)

![Normalized confusion matrix showing per-class prediction accuracy](assets/training/confusion_matrix.png)

![Per-class mAP50-95 comparison across models](assets/comparison/per_class_mAP_comparison.png)

---

## Performance Benchmarks

### Expected Results on BFMC Dataset

| Model | mAP50 | mAP50-95 | FPS (4090) | Size |
|-------|-------|----------|------------|------|
| YOLOv8n | ~0.85 | ~0.60 | ~800 | 6 MB |
| YOLOv11n | ~0.87 | ~0.63 | ~750 | 5 MB |
| RT-DETR-l | ~0.89 | ~0.68 | ~150 | 65 MB |

### Inference Speed (640x640 input)

| Hardware | YOLOv8n | YOLOv11n | RT-DETR-l |
|----------|---------|----------|-----------|
| RTX 4090 | 800 FPS | 750 FPS | 150 FPS |
| RTX 3090 | 600 FPS | 550 FPS | 120 FPS |
| RTX 3060 | 400 FPS | 380 FPS | 80 FPS |
| Jetson Xavier | 120 FPS | 110 FPS | 25 FPS |
| Raspberry Pi 4 | 15 FPS | 14 FPS | 3 FPS |

---

## Hardware Requirements

### For Training

**Minimum**:

- CPU: 4+ cores
- RAM: 16 GB
- GPU: NVIDIA GPU with 8GB VRAM
- Storage: 10 GB

**Recommended**:

- CPU: 8+ cores
- RAM: 32 GB
- GPU: RTX 3090/4090 (24GB VRAM)
- Storage: 50 GB (SSD)

### Deployment

**Raspberry Pi 4/5**:

- Use YOLOv8n with ONNX export
- Expected FPS: ~15

**Jetson Xavier/Orin**:

- Use YOLOv8n/s with TensorRT
- Expected FPS: ~60-120

---

## Deployment Notes

### For Raspberry Pi 4/5

1. Use nano/small model variants
2. Export to ONNX format
3. Consider OpenVINO for acceleration

### For Jetson Nano/Xavier

1. Export to TensorRT engine
2. Use FP16 precision
3. nano/small variants recommended

### For Competition

1. Test inference speed on target hardware
2. Ensure >30 FPS for real-time operation
3. Validate on representative test scenarios

---

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size in `training/configs/config.yaml`
- Use smaller model variant (nano instead of small)
- Disable cache: `cache: false`

### Slow Training

- Enable `cache: true` in config
- Ensure `amp: true` is set
- Check GPU utilization with `nvidia-smi`

### Poor Accuracy

- Increase epochs (try 150-200)
- Try larger model variant
- Adjust augmentation settings
- Verify dataset quality with `check_split_quality.py`

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Ultralytics** - YOLO implementation
- **Roboflow** - Dataset hosting and tools
- **BFMC Team** - Competition organization
- **Team DriverIES** - Dataset creation

---
