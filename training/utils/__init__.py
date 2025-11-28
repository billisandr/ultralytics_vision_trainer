"""BFMC Vision Training Utilities"""

from .data import (
    analyze_dataset,
    validate_labels,
    get_class_distribution,
    check_image_quality,
)

from .visualize import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_class_distribution,
    plot_model_comparison,
    visualize_predictions,
)

from .video import (
    extract_frames,
    annotate_video,
    create_tracking_video,
    get_video_info,
)

__all__ = [
    # Data utilities
    'analyze_dataset',
    'validate_labels',
    'get_class_distribution',
    'check_image_quality',
    # Visualization
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_class_distribution',
    'plot_model_comparison',
    'visualize_predictions',
    # Video processing
    'extract_frames',
    'annotate_video',
    'create_tracking_video',
    'get_video_info',
]