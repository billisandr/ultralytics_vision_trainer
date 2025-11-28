#!/usr/bin/env python3
"""
Video processing utilities for BFMC Vision.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from tqdm import tqdm


def extract_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 1,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None
) -> List[str]:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        frame_interval: Extract every Nth frame
        max_frames: Maximum number of frames to extract
        resize: Optional (width, height) to resize frames

    Returns:
        List of saved frame paths
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps:.1f}")

    saved_paths = []
    frame_count = 0
    saved_count = 0

    with tqdm(total=min(total_frames, max_frames or total_frames)) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                if resize:
                    frame = cv2.resize(frame, resize)

                frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_paths.append(frame_path)
                saved_count += 1
                pbar.update(1)

                if max_frames and saved_count >= max_frames:
                    break

            frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")

    return saved_paths


def annotate_video(
    video_path: str,
    output_path: str,
    model,
    config: dict,
    show_fps: bool = True,
    show_labels: bool = True
):
    """
    Annotate a video with object detection results.

    Args:
        video_path: Path to input video
        output_path: Path to save annotated video
        model: Loaded detection model
        config: Configuration dictionary
        show_fps: Show FPS overlay
        show_labels: Show class labels
    """
    import time

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Annotating video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.1f}")

    fps_list = []

    with tqdm(total=total_frames) as pbar:
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
                verbose=False,
            )
            inference_time = time.time() - start_time
            current_fps = 1.0 / inference_time
            fps_list.append(current_fps)

            # Draw results
            annotated = results[0].plot()

            # Add FPS overlay
            if show_fps:
                cv2.putText(
                    annotated,
                    f"FPS: {current_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            out.write(annotated)
            pbar.update(1)

    cap.release()
    out.release()

    avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"Saved annotated video to: {output_path}")
    print(f"Average inference FPS: {avg_fps:.1f}")


def track_objects_sam2(
    video_path: str,
    output_dir: str,
    sam_model,
    initial_boxes: List[List[float]],
    initial_frame: int = 0,
    use_boxes: bool = True
) -> Dict:
    """
    DEPRECATED: SAM2 support has been removed from this project.
    Track objects in video using SAM2.

    Args:
        video_path: Path to input video
        output_dir: Directory to save tracking results
        sam_model: Loaded SAM2 model
        initial_boxes: List of bounding boxes [x1, y1, x2, y2] for objects to track
        initial_frame: Frame number with initial boxes
        use_boxes: Use bounding boxes as prompts (vs center points)

    Returns:
        Dictionary with tracking results
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    tracking_results = {
        'video': video_path,
        'num_objects': len(initial_boxes),
        'fps': fps,
        'total_frames': total_frames,
        'frames': {},
    }

    # Generate colors for each object
    np.random.seed(42)
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
              for _ in range(len(initial_boxes))]

    print(f"Tracking {len(initial_boxes)} objects through {total_frames} frames...")

    # Previous masks for propagation (simple tracking approach)
    prev_boxes = [list(box) for box in initial_boxes]

    with tqdm(total=total_frames) as pbar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Prepare prompts
            if use_boxes:
                # Use bounding boxes as prompts
                bboxes = np.array(prev_boxes)
                results = sam_model.predict(
                    source=frame,
                    bboxes=bboxes,
                    verbose=False,
                )
            else:
                # Use center points as prompts
                points = []
                for box in prev_boxes:
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    points.append([cx, cy])
                points = np.array(points)
                labels = np.ones(len(points))

                results = sam_model.predict(
                    source=frame,
                    points=points,
                    labels=labels,
                    verbose=False,
                )

            # Extract masks and update tracking
            frame_data = {'masks': [], 'boxes': []}

            if results and len(results) > 0:
                result = results[0]
                masks = result.masks

                if masks is not None:
                    # Save mask visualization
                    mask_path = os.path.join(output_dir, f"mask_{frame_idx:06d}.png")

                    # Create colored mask overlay
                    combined = np.zeros((*frame.shape[:2], 3), dtype=np.uint8)

                    for i, mask in enumerate(masks.data):
                        color = colors[i % len(colors)]
                        mask_np = mask.cpu().numpy().astype(bool)
                        combined[mask_np] = color

                        # Update bounding box from mask for next frame
                        if mask_np.any():
                            ys, xs = np.where(mask_np)
                            new_box = [xs.min(), ys.min(), xs.max(), ys.max()]
                            if i < len(prev_boxes):
                                prev_boxes[i] = new_box
                            frame_data['boxes'].append(new_box)

                        frame_data['masks'].append(mask_path)

                    cv2.imwrite(mask_path, combined)

                    tracking_results['frames'][frame_idx] = {
                        'mask_path': mask_path,
                        'num_masks': len(masks.data),
                        'boxes': frame_data['boxes'],
                    }

            frame_idx += 1
            pbar.update(1)

    cap.release()

    # Save tracking results
    import json
    results_path = os.path.join(output_dir, "tracking_results.json")
    with open(results_path, 'w') as f:
        json.dump(tracking_results, f, indent=2, default=str)

    print(f"Tracking results saved to: {output_dir}")

    return tracking_results


def detect_and_track(
    video_path: str,
    output_dir: str,
    detection_model,
    sam_model,
    config: dict,
    classes_to_track: Optional[List[int]] = None
) -> Dict:
    """
    DEPRECATED: SAM2 support has been removed from this project.
    Detect objects in first frame and track them with SAM2.

    Args:
        video_path: Path to input video
        output_dir: Directory to save results
        detection_model: YOLO/RT-DETR model for initial detection
        sam_model: SAM2 model for tracking
        config: Configuration dictionary
        classes_to_track: Optional list of class IDs to track

    Returns:
        Dictionary with tracking results
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Read first frame
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Could not read first frame")

    # Detect objects
    results = detection_model.predict(
        source=frame,
        conf=config['inference']['conf_threshold'],
        iou=config['inference']['iou_threshold'],
        verbose=False,
    )

    if not results or len(results) == 0:
        return {'error': 'No objects detected'}

    # Extract boxes
    boxes = results[0].boxes
    initial_boxes = []
    detected_classes = []

    for box in boxes:
        cls_id = int(box.cls[0])
        if classes_to_track is None or cls_id in classes_to_track:
            xyxy = box.xyxy[0].tolist()
            initial_boxes.append(xyxy)
            detected_classes.append(cls_id)

    if not initial_boxes:
        return {'error': 'No matching objects detected'}

    print(f"Detected {len(initial_boxes)} objects to track")

    # Track with SAM2
    tracking_results = track_objects_sam2(
        video_path,
        output_dir,
        sam_model,
        initial_boxes,
    )

    tracking_results['detected_classes'] = detected_classes

    return tracking_results


def create_tracking_video(
    video_path: str,
    masks_dir: str,
    output_path: str,
    alpha: float = 0.5
):
    """
    Create video with tracking masks overlay.

    Args:
        video_path: Path to original video
        masks_dir: Directory with mask images
        output_path: Path to save output video
        alpha: Mask overlay transparency
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Creating tracking video: {output_path}")

    with tqdm(total=total_frames) as pbar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Load mask if exists
            mask_path = os.path.join(masks_dir, f"mask_{frame_idx:06d}.png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path)
                if mask is not None:
                    # Resize mask if needed
                    if mask.shape[:2] != frame.shape[:2]:
                        mask = cv2.resize(mask, (width, height))

                    # Blend mask with frame
                    frame = cv2.addWeighted(frame, 1 - alpha, mask, alpha, 0)

            out.write(frame)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    out.release()

    print(f"Saved tracking video to: {output_path}")


def get_video_info(video_path: str) -> Dict:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': f'Could not open video: {video_path}'}

    info = {
        'path': video_path,
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration_s': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    cap.release()
    return info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Video processing utilities")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--action", type=str, choices=['info', 'extract'], default='info')
    parser.add_argument("--output", type=str, default="frames", help="Output directory")
    parser.add_argument("--interval", type=int, default=1, help="Frame interval")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames")

    args = parser.parse_args()

    if args.action == 'info':
        info = get_video_info(args.video)
        for key, value in info.items():
            print(f"{key}: {value}")
    elif args.action == 'extract':
        extract_frames(
            args.video,
            args.output,
            frame_interval=args.interval,
            max_frames=args.max_frames
        )