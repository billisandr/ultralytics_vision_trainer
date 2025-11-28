#!/usr/bin/env python3
"""
Check if your dataset split is adequate.
Run this after training to see if you need to rebalance.
"""

import sys
from pathlib import Path

def analyze_split_quality(results_dir):
    """Analyze if validation/test splits are adequate."""
    
    # Load results.csv
    import pandas as pd
    results_path = Path(results_dir) / "results.csv"
    
    if not results_path.exists():
        print(f"[ERROR] Results file not found: {results_path}")
        return
    
    df = pd.read_csv(results_path)
    
    # Calculate validation metric variance
    if 'metrics/mAP50(B)' in df.columns:
        val_map = df['metrics/mAP50(B)'].dropna()
        val_variance = val_map.std()
        val_mean = val_map.mean()
        
        print("=" * 60)
        print("VALIDATION SET QUALITY ANALYSIS")
        print("=" * 60)
        print(f"Mean validation mAP50: {val_mean:.4f}")
        print(f"Std deviation: {val_variance:.4f}")
        print(f"Coefficient of variation: {val_variance/val_mean:.2%}")
        print()
        
        if val_variance / val_mean > 0.05:
            print("[WARN] HIGH VARIANCE in validation metrics!")
            print("   Your validation set might be too small.")
            print("   Consider increasing validation split to 10%")
        else:
            print("[OK] Validation metrics are stable")
            print("   Current validation split size is adequate")
    
    print()
    print("RECOMMENDATION:")
    print("-" * 60)
    
    # Check epoch count
    total_epochs = len(df)
    if total_epochs < 50:
        print("[WARN] Train for more epochs first (need 50+ for analysis)")
    else:
        if val_variance / val_mean > 0.05:
            print("[INFO] Consider rebalancing to 80/10/10 or 85/7.5/7.5")
            print("   This will give more reliable validation metrics")
        else:
            print("[OK] Keep current split - it's working well!")
            print("   Your validation set is adequately sized")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_split_quality(sys.argv[1])
    else:
        print("Usage: python check_split_quality.py <results_dir>")
        print("Example: python check_split_quality.py results/yolov8_20241125_120000")
