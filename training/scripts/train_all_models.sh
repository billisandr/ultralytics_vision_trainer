#!/bin/bash
# Train all three models with comprehensive monitoring
# This script trains YOLOv8, YOLOv11, and RT-DETR sequentially

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(dirname "$SCRIPT_DIR")"
cd "$TRAINING_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
EPOCHS=${1:-100}
BATCH_SIZE=${2:-16}
LOG_DIR="logs"
RESULTS_DIR="results"

# Create log directory
mkdir -p "$LOG_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}BFMC Vision: Train All Models${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Log Directory: $LOG_DIR"
echo ""

# Models to train
MODELS=("yolov8" "yolov11" "rtdetr")
MODEL_NAMES=("YOLOv8" "YOLOv11" "RT-DETR")

# Estimate times (RTX 4090)
TIMES=("~45 min" "~45 min" "~2 hours")

echo -e "${YELLOW}Models to train:${NC}"
for i in "${!MODELS[@]}"; do
    echo "  $((i+1)). ${MODEL_NAMES[$i]} (${TIMES[$i]})"
done
echo ""

# Calculate total time
echo -e "${YELLOW}Estimated total time: ~3-4 hours (RTX 4090)${NC}"
echo ""

# Ask for confirmation
read -p "Start training all models? (yes/no): " -r
echo
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo -e "${RED}Training cancelled${NC}"
    exit 0
fi

# Function to train a single model
train_model() {
    local model=$1
    local model_name=$2
    local model_num=$3
    local total_models=$4

    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}[$model_num/$total_models] Training $model_name${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # Log file
    local log_file="$LOG_DIR/${model}_$(date +%Y%m%d_%H%M%S).log"
    local start_time=$(date +%s)

    # Train with output to both console and log
    echo -e "${GREEN}Starting training...${NC}"
    echo "Log file: $log_file"
    echo ""

    python3 scripts/train.py \
        --model "$model" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        2>&1 | tee "$log_file"

    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_min=$((duration / 60))

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo -e "${GREEN}[OK] $model_name training completed!${NC}"
        echo -e "${GREEN}   Duration: ${duration_min} minutes${NC}"
        echo -e "${GREEN}   Log: $log_file${NC}"

        # Find results directory
        local result_dir=$(ls -td results/${model}_* 2>/dev/null | head -1)
        if [ -n "$result_dir" ]; then
            echo -e "${GREEN}   Results: $result_dir${NC}"

            # Show final metrics if available
            if [ -f "$result_dir/results.csv" ]; then
                echo -e "${GREEN}   Final metrics:${NC}"
                tail -1 "$result_dir/results.csv" | awk -F',' '{
                    printf "     mAP50: %.3f | mAP50-95: %.3f\n", $13, $14
                }'
            fi
        fi
        echo ""
        return 0
    else
        echo ""
        echo -e "${RED}[ERROR] $model_name training failed!${NC}"
        echo -e "${RED}   Check log: $log_file${NC}"
        echo ""
        return 1
    fi
}

# Train each model
START_TIME=$(date +%s)
FAILED_MODELS=()

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    model_num=$((i+1))
    total_models=${#MODELS[@]}

    if ! train_model "$model" "$model_name" "$model_num" "$total_models"; then
        FAILED_MODELS+=("$model_name")

        # Ask if user wants to continue
        echo -e "${YELLOW}Continue with remaining models? (yes/no): ${NC}"
        read -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            echo -e "${RED}Training stopped${NC}"
            break
        fi
    fi

    # Short pause between models
    if [ $((i+1)) -lt $total_models ]; then
        echo -e "${BLUE}Next model starts in 10 seconds...${NC}"
        sleep 10
    fi
done

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINS=$(((TOTAL_DURATION % 3600) / 60))

# Summary
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TRAINING SUMMARY${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Total Duration: ${TOTAL_HOURS}h ${TOTAL_MINS}m"
echo ""

if [ ${#FAILED_MODELS[@]} -eq 0 ]; then
    echo -e "${GREEN}[OK] All models trained successfully!${NC}"
else
    echo -e "${YELLOW}[WARN]  Some models failed:${NC}"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  - $model"
    done
fi

echo ""
echo "Logs saved in: $LOG_DIR"
echo "Results saved in: $RESULTS_DIR"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Compare models:"
echo "     python3 scripts/evaluate.py"
echo ""
echo "  2. View TensorBoard:"
echo "     tensorboard --logdir results"
echo ""
echo "  3. Check individual results:"
echo "     ls -lh results/*/weights/best.pt"
echo ""
