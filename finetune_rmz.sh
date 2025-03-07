#!/bin/bash

# Function to print usage
print_usage() {
    echo "Usage:"
    echo "  $0 new                          # Start new training"
    echo "  $0 continue <existing_run_dir>  # Continue training from existing directory"
}

# Base directories
BASE_DIR="/home/twbgmy/play/marma-tts/mms-rmz"
SCRIPT_DIR="/home/twbgmy/play/marma-tts/finetune-hf-vits-marma"

if [ "$1" = "new" ]; then
    # Create new training session
    TIMESTAMP=$(date +%Y-%m-%d_%H-%M)
    RUN_DIR="${BASE_DIR}/finetuned/run_${TIMESTAMP}"
    
    # Create run directory
    mkdir -p "${RUN_DIR}/logs"
    
    # Copy and modify config file
    cp "${SCRIPT_DIR}/finetune_mms_rmz.json" "${RUN_DIR}/finetune_mms_rmz.json"
    sed -i "s|\"output_dir\": \".*\"|\"output_dir\": \"${RUN_DIR}\"|" "${RUN_DIR}/finetune_mms_rmz.json"

elif [ "$1" = "continue" ]; then
    if [ -z "$2" ]; then
        echo "Error: Please provide the existing run directory to continue training from"
        print_usage
        exit 1
    fi
    
    RUN_DIR="$2"
    
    if [ ! -d "$RUN_DIR" ]; then
        echo "Error: Directory $RUN_DIR does not exist"
        exit 1
    fi
    
    # Find latest checkpoint
    LATEST_CHECKPOINT=$(ls -d ${RUN_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "Warning: No checkpoints found in $RUN_DIR"
    else
        echo "Continuing from checkpoint: $LATEST_CHECKPOINT"
        # Add resume_from_checkpoint to the JSON config
        sed -i "s|{|{\n  \"resume_from_checkpoint\": \"${LATEST_CHECKPOINT}\",|" "${RUN_DIR}/finetune_mms_rmz.json"
    fi

else
    print_usage
    exit 1
fi

# Start training
echo "Training in directory: $RUN_DIR"
accelerate launch --num_processes=1 \
    "${SCRIPT_DIR}/run_vits_finetuning.py" \
    "${RUN_DIR}/finetune_mms_rmz.json" 2>&1 | tee -a "${RUN_DIR}/logs/training.log"

    