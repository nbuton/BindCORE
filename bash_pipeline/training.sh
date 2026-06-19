#!/bin/bash

# Define the base directory for the models
MODELS_DIR="data/models"

# Check if the models directory actually exists before starting
if [ ! -d "$MODELS_DIR" ]; then
    echo "Error: Directory $MODELS_DIR not found."
    echo "Please ensure you are running this script from the ~/BindCORE root directory."
    exit 1
fi

echo "Starting sequential model training..."
echo "------------------------------------------------"

# Loop through every item inside data/models
for model_path in "$MODELS_DIR"/*; do
    
    # Ensure we are looking at a directory, not a stray file
    if [ -d "$model_path" ]; then
        model_name=$(basename "$model_path")
        config_path="$model_path/config.yaml"
        
        # Verify the config.yaml file exists for this specific model
        if [ -f "$config_path" ]; then
            echo "[STARTING] Training for model: $model_name"
            echo "Running: python3 scripts/train.py --config $config_path --device cuda"
            
            # Execute training. 
            # If it exits with an error (non-zero), the code inside || { ... } runs,
            # but the loop itself keeps chugging along.
            python3 scripts/train.py --config "$config_path" --device cuda || {
                echo "[CRASH DETECTED] $model_name failed! Skipping to next..."
            }
            
            echo "[FINISHED/SKIPPED] Process for $model_name complete."
            echo "------------------------------------------------"
        else
            echo "[SKIP] No config.yaml found in $model_path"
            echo "------------------------------------------------"
        fi
    fi
done

echo "All model training loops have been processed!"