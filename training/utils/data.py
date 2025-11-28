#!/usr/bin/env python3
"""
Dataset analysis and validation utilities for BFMC Vision.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
from PIL import Image


def load_data_yaml(data_yaml_path: str) -> dict:
    """Load and parse data.yaml file."""
    with open(data_yaml_path, 'r') as f:
        return yaml.safe_load(f)


def analyze_dataset(data_yaml_path: str) -> Dict:
    """
    Analyze dataset and return comprehensive statistics.

    Args:
        data_yaml_path: Path to data.yaml file

    Returns:
        Dictionary with dataset statistics
    """
    config = load_data_yaml(data_yaml_path)
    base_path = Path(data_yaml_path).parent

    stats = {
        'num_classes': config.get('nc', len(config.get('names', []))),
        'class_names': config.get('names', []),
        'splits': {},
        'total_images': 0,
        'total_annotations': 0,
    }

    # Analyze each split
    for split in ['train', 'val', 'test']:
        split_path = config.get(split)
        if split_path:
            # Handle relative paths
            if not os.path.isabs(split_path):
                split_path = base_path / split_path
            else:
                split_path = Path(split_path)

            # Get images directory
            if split_path.name == 'images':
                images_dir = split_path
            else:
                images_dir = split_path / 'images'

            labels_dir = images_dir.parent / 'labels'

            if images_dir.exists():
                images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
                num_images = len(images)

                # Count annotations
                num_annotations = 0
                if labels_dir.exists():
                    for label_file in labels_dir.glob('*.txt'):
                        with open(label_file, 'r') as f:
                            num_annotations += len(f.readlines())

                stats['splits'][split] = {
                    'images': num_images,
                    'annotations': num_annotations,
                    'avg_annotations_per_image': num_annotations / max(num_images, 1),
                }
                stats['total_images'] += num_images
                stats['total_annotations'] += num_annotations

    return stats


def get_class_distribution(data_yaml_path: str, split: str = 'train') -> Dict[str, int]:
    """
    Get class distribution for a dataset split.

    Args:
        data_yaml_path: Path to data.yaml file
        split: Dataset split ('train', 'val', 'test')

    Returns:
        Dictionary mapping class names to counts
    """
    config = load_data_yaml(data_yaml_path)
    base_path = Path(data_yaml_path).parent
    class_names = config.get('names', [])

    # Get labels directory
    split_path = config.get(split)
    if not split_path:
        return {}

    if not os.path.isabs(split_path):
        split_path = base_path / split_path
    else:
        split_path = Path(split_path)

    if split_path.name == 'images':
        labels_dir = split_path.parent / 'labels'
    else:
        labels_dir = split_path / 'labels'

    # Count classes
    class_counts = Counter()

    if labels_dir.exists():
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        if class_id < len(class_names):
                            class_counts[class_names[class_id]] += 1
                        else:
                            class_counts[f'class_{class_id}'] += 1

    return dict(class_counts)


def validate_labels(data_yaml_path: str, split: str = 'train') -> Dict:
    """
    Validate label files for common issues.

    Args:
        data_yaml_path: Path to data.yaml file
        split: Dataset split to validate

    Returns:
        Dictionary with validation results
    """
    config = load_data_yaml(data_yaml_path)
    base_path = Path(data_yaml_path).parent
    num_classes = config.get('nc', len(config.get('names', [])))

    # Get paths
    split_path = config.get(split)
    if not split_path:
        return {'error': f'Split {split} not found'}

    if not os.path.isabs(split_path):
        split_path = base_path / split_path
    else:
        split_path = Path(split_path)

    if split_path.name == 'images':
        images_dir = split_path
    else:
        images_dir = split_path / 'images'

    labels_dir = images_dir.parent / 'labels'

    results = {
        'total_labels': 0,
        'valid_labels': 0,
        'issues': [],
        'missing_labels': [],
        'invalid_class_ids': [],
        'invalid_coordinates': [],
    }

    # Check each image has a label
    images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

    for img_path in images:
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            results['missing_labels'].append(str(img_path.name))
            continue

        results['total_labels'] += 1
        is_valid = True

        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue

                # Check format
                if len(parts) < 5:
                    results['issues'].append(f"{label_path.name}:{line_num} - Invalid format")
                    is_valid = False
                    continue

                try:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]

                    # Check class ID
                    if class_id < 0 or class_id >= num_classes:
                        results['invalid_class_ids'].append(
                            f"{label_path.name}:{line_num} - class {class_id}"
                        )
                        is_valid = False

                    # Check coordinates (should be 0-1)
                    for coord in coords:
                        if coord < 0 or coord > 1:
                            results['invalid_coordinates'].append(
                                f"{label_path.name}:{line_num} - {coords}"
                            )
                            is_valid = False
                            break

                except ValueError as e:
                    results['issues'].append(f"{label_path.name}:{line_num} - {e}")
                    is_valid = False

        if is_valid:
            results['valid_labels'] += 1

    return results


def check_image_quality(data_yaml_path: str, split: str = 'train', sample_size: int = 100) -> Dict:
    """
    Check image quality metrics for a dataset split.

    Args:
        data_yaml_path: Path to data.yaml file
        split: Dataset split to check
        sample_size: Number of images to sample

    Returns:
        Dictionary with image quality statistics
    """
    config = load_data_yaml(data_yaml_path)
    base_path = Path(data_yaml_path).parent

    # Get images directory
    split_path = config.get(split)
    if not split_path:
        return {'error': f'Split {split} not found'}

    if not os.path.isabs(split_path):
        split_path = base_path / split_path
    else:
        split_path = Path(split_path)

    if split_path.name == 'images':
        images_dir = split_path
    else:
        images_dir = split_path / 'images'

    images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

    # Sample images
    if len(images) > sample_size:
        import random
        images = random.sample(images, sample_size)

    widths = []
    heights = []
    file_sizes = []
    formats = Counter()
    corrupted = []

    for img_path in images:
        try:
            with Image.open(img_path) as img:
                widths.append(img.width)
                heights.append(img.height)
                formats[img.format] += 1
            file_sizes.append(os.path.getsize(img_path))
        except Exception as e:
            corrupted.append(str(img_path.name))

    if not widths:
        return {'error': 'No valid images found'}

    return {
        'sampled': len(images),
        'width': {
            'min': min(widths),
            'max': max(widths),
            'mean': np.mean(widths),
            'std': np.std(widths),
        },
        'height': {
            'min': min(heights),
            'max': max(heights),
            'mean': np.mean(heights),
            'std': np.std(heights),
        },
        'file_size_kb': {
            'min': min(file_sizes) / 1024,
            'max': max(file_sizes) / 1024,
            'mean': np.mean(file_sizes) / 1024,
        },
        'formats': dict(formats),
        'corrupted': corrupted,
    }


def print_dataset_summary(data_yaml_path: str):
    """Print a formatted summary of the dataset."""
    stats = analyze_dataset(data_yaml_path)

    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)

    print(f"\nClasses ({stats['num_classes']}):")
    for i, name in enumerate(stats['class_names']):
        print(f"  {i}: {name}")

    print(f"\nTotal images: {stats['total_images']}")
    print(f"Total annotations: {stats['total_annotations']}")

    print("\nSplits:")
    for split, info in stats['splits'].items():
        print(f"  {split}:")
        print(f"    Images: {info['images']}")
        print(f"    Annotations: {info['annotations']}")
        print(f"    Avg per image: {info['avg_annotations_per_image']:.1f}")

    # Class distribution
    print("\nClass distribution (train):")
    dist = get_class_distribution(data_yaml_path, 'train')
    for class_name, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {class_name}: {count}")

    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze BFMC dataset")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--validate", action="store_true", help="Validate labels")
    parser.add_argument("--quality", action="store_true", help="Check image quality")

    args = parser.parse_args()

    print_dataset_summary(args.data)

    if args.validate:
        print("\nValidating labels...")
        results = validate_labels(args.data)
        print(f"Valid: {results['valid_labels']}/{results['total_labels']}")
        if results['missing_labels']:
            print(f"Missing labels: {len(results['missing_labels'])}")
        if results['issues']:
            print(f"Issues: {len(results['issues'])}")

    if args.quality:
        print("\nChecking image quality...")
        quality = check_image_quality(args.data)
        print(f"Resolution: {quality['width']['mean']:.0f}x{quality['height']['mean']:.0f}")
        print(f"File size: {quality['file_size_kb']['mean']:.1f} KB avg")