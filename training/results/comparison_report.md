# BFMC Vision Model Comparison Report
Generated: 2025-11-25 21:43:19

## Performance Summary

| Model | Type | mAP50 | Rating | mAP50-95 | Rating | Precision | Recall | FPS | Size (MB) |
|-------|------|-------|--------|----------|--------|-----------|--------|-----|----------|
| yolov8_yolov8n_20251125_122425 | trained | 0.975 | Excellent | 0.881 | Excellent | 0.963 | 0.965 | 378.7 | 6.0 |
| yolov11_yolo11n_20251125_150336 | trained | 0.975 | Excellent | 0.874 | Excellent | 0.947 | 0.982 | 305.2 | 5.2 |
| rtdetr_rtdetr-l_20251125_183331 | trained | 0.977 | Excellent | 0.867 | Excellent | 0.942 | 0.980 | 81.5 | 63.2 |

### Rating Scale

**mAP50-95 Ratings:**
- Excellent: ≥0.70 | Very Good: ≥0.60 | Good: ≥0.50 | Fair: ≥0.40 | Needs Improvement: ≥0.30 | Poor: <0.30

**mAP50 Ratings:**
- Excellent: ≥0.85 | Very Good: ≥0.75 | Good: ≥0.65 | Fair: ≥0.55 | Needs Improvement: ≥0.45 | Poor: <0.45

## Per-Class Performance

Detailed per-class analysis showing which models perform best on each object class.

### AP50 (IoU=0.5)

![Per-Class AP50 Comparison](per_class_ap50_comparison.png)

*Grouped bar chart comparing AP50 across all classes*

![Per-Class AP50 Heatmap](per_class_ap50_heatmap.png)

*Heatmap showing AP50 performance patterns across models and classes*

### AP50-95 (IoU=0.5:0.95)

![Per-Class AP50-95 Comparison](per_class_ap50-95_comparison.png)

*Grouped bar chart comparing AP50-95 across all classes*

![Per-Class AP50-95 Heatmap](per_class_ap50-95_heatmap.png)

*Heatmap showing AP50-95 performance patterns across models and classes*


## Recommendations

- **Best Accuracy**: yolov8_yolov8n_20251125_122425 (mAP50-95: 0.881)
- **Fastest Inference**: yolov8_yolov8n_20251125_122425 (378.7 FPS)
- **Smallest Size**: yolov11_yolo11n_20251125_150336 (5.2 MB)

## Trade-off Analysis

For BFMC competition, consider:

1. **Real-time requirement**: Need >30 FPS for smooth operation
2. **Raspberry Pi deployment**: Smaller models preferred
3. **Detection accuracy**: Critical for traffic signs and pedestrians
