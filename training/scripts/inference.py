"""
BFMC Vision Inference Script
Run inference on images, videos, or camera streams.

Capabilities:
1. Run Inference: Detect objects in images, videos, or a live camera stream
2. Export Models: Convert .pt models to ONNX, TensorRT, or TFLite
3. Visualize: Display results with bounding boxes or save them to disk

Usage Examples:
    python3 scripts/inference.py --weights results/yolov8_best.pt
    python3 scripts/inference.py --weights results/yolov8_best.pt --camera 0
    python3 scripts/inference.py --weights results/yolov8_best.pt --source video.mp4 --save
    python3 scripts/inference.py --weights results/yolov8_best.pt --export onnx
"""

import argparse
import yaml
import cv2
import time
from pathlib import Path
from ultralytics import YOLO, RTDETR


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(weights_path: str):
    """Load model from weights file."""
    if "rtdetr" in weights_path.lower():
        return RTDETR(weights_path)
    return YOLO(weights_path)


def run_inference(
    model,
    source,
    config: dict,
    save: bool = False,
    show: bool = True,
    output_dir: str = "results/inference"
):
    """Run inference on source."""
    results = model.predict(
        source=source,
        conf=config['inference']['conf_threshold'],
        iou=config['inference']['iou_threshold'],
        max_det=config['inference']['max_det'],
        imgsz=config['training']['imgsz'],
        device=config['training']['device'],
        save=save,
        show=show,
        project=output_dir,
        name="detect",
        stream=True,  # Generator for video/stream
    )

    # Process results
    for result in results:
        if result.boxes is not None:
            # Get detections
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                class_name = result.names[cls]
                print(f"  {class_name}: {conf:.2f} at {xyxy}")


def run_camera_stream(
    model,
    config: dict,
    camera_id: int = 0,
    output_dir: str = "results/inference"
):
    """Run real-time inference on camera stream."""
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    print(f"Running inference on camera {camera_id}. Press 'q' to quit.")

    fps_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        start_time = time.time()
        results = model.predict(
            source=frame,
            conf=config['inference']['conf_threshold'],
            iou=config['inference']['iou_threshold'],
            max_det=config['inference']['max_det'],
            imgsz=config['training']['imgsz'],
            device=config['training']['device'],
            verbose=False,
        )
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time
        fps_list.append(fps)

        # Draw results
        annotated_frame = results[0].plot()

        # Add FPS overlay
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display
        cv2.imshow("BFMC Vision", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print average FPS
    if fps_list:
        avg_fps = sum(fps_list) / len(fps_list)
        print(f"\nAverage FPS: {avg_fps:.1f}")


def export_model(model, weights_path: str, format: str = "onnx"):
    """Export model to different formats for deployment."""
    print(f"Exporting model to {format} format...")

    model.export(
        format=format,
        imgsz=640,
        half=True,  # FP16 for faster inference
        simplify=True,  # Simplify ONNX model
    )

    output_path = Path(weights_path).with_suffix(f".{format}")
    print(f"Model exported to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="BFMC Vision Inference")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to model weights"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Image/video path or camera ID"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera ID for live inference"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save inference results"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display results"
    )
    parser.add_argument(
        "--export",
        type=str,
        choices=["onnx", "tflite", "engine", "openvino"],
        default=None,
        help="Export model to format"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/inference",
        help="Output directory"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Load model
    print(f"Loading model: {args.weights}")
    model = load_model(args.weights)

    # Export if requested
    if args.export:
        export_model(model, args.weights, args.export)
        return

    # Run inference
    if args.camera is not None:
        run_camera_stream(model, config, args.camera, args.output)
    elif args.source:
        run_inference(
            model,
            args.source,
            config,
            save=args.save,
            show=not args.no_show,
            output_dir=args.output
        )
    else:
        # Default: run on test set
        test_path = Path(config['dataset']['test']) / "images"
        print(f"Running inference on test set: {test_path}")
        run_inference(
            model,
            str(test_path),
            config,
            save=args.save,
            show=not args.no_show,
            output_dir=args.output
        )


if __name__ == "__main__":
    main()
