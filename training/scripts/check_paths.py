#!/usr/bin/env python3
"""
Check Paths Script
Verifies how paths are generated and resolved in the training setup.
"""

import yaml
import os
from pathlib import Path
import sys

def check_paths():
    print(f"\n{'='*60}")
    print("Checking BFMC Vision Paths")
    print(f"{'='*60}\n")
    
    # 1. Determine Context
    cwd = Path.cwd()
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    training_root = script_path.parent.parent # bfmc_vision_simple/training
    project_root = training_root.parent
    
    print(f"Current Working Directory: {cwd}")
    print(f"Script Location:           {script_path}")
    print(f"Training Root:             {training_root}")
    print(f"Project Root:              {project_root}")
    print("-" * 60)

    # 2. Config Path Resolution (mimicking train.py)
    # train.py defaults to: str(Path(__file__).parent.parent / "configs" / "config.yaml")
    config_path = training_root / "configs" / "config.yaml"
    print(f"Config File Path:          {config_path}")
    
    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}")
        return

    # 3. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("-" * 60)
    
    # 4. Output Paths (mimicking train.py logic)
    print("OUTPUT PATHS (Resolved relative to training/ directory):")
    # train.py logic:
    # script_dir = Path(__file__).parent.parent
    # results_dir = script_dir / config['output']['results_dir']
    
    raw_results_dir = config['output']['results_dir']
    models_dir_raw = config['output']['models_dir']
    
    resolved_results_dir = training_root / raw_results_dir
    resolved_models_dir = training_root / models_dir_raw
    
    print(f"  Results Dir (Raw):       {raw_results_dir}")
    print(f"  Results Dir (Resolved):  {resolved_results_dir}")
    print(f"  Models Dir (Raw):        {models_dir_raw}")
    print(f"  Models Dir (Resolved):   {resolved_models_dir}")
    
    print("-" * 60)

    # 5. Dataset Paths
    print("DATASET PATHS:")
    dataset_conf = config.get('dataset', {})
    
    for key, val in dataset_conf.items():
        # These paths are tricky. They are usually relative to CWD if passed to YOLO?
        # Or relative to the data.yaml if inside data.yaml?
        # But here they are in config.yaml.
        # Let's check resolution relative to CWD and Training Root.
        
        print(f"  {key}: {val}")
        
        # Check relative to CWD
        path_cwd = Path(val).resolve()
        exists_cwd = path_cwd.exists() or (path_cwd.parent.exists() and not path_cwd.is_dir()) # strict check might fail for non-existant output
        
        # Check relative to Training Root (likely intention if running from training/)
        path_training = (training_root / val).resolve()
        exists_training = path_training.exists()
        
        # Check relative to Project Root (if running from root)
        path_root = (project_root / val).resolve()
        exists_root = path_root.exists()

        status = []
        if exists_cwd: status.append(f"Found via CWD")
        if exists_training: status.append(f"Found via training/ ({path_training})")
        if exists_root: status.append(f"Found via project root ({path_root})")
        
        if not status:
            print(f"    WARNING: Path does NOT exist based on CWD, training/, or project root.")
            print(f"    - tried CWD: {path_cwd}")
            print(f"    - tried training/: {path_training}")
            print(f"    - tried project root: {path_root}")
        else:
            print(f"    STATUS: {', '.join(status)}")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    check_paths()
