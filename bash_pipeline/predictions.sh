#!/bin/bash

# Configuration paths
MODELS_DIR="data/models"
OUTPUT_DIR="data/predictions"

# Based on your note: "the complete is the less than 1024" for LIP
LIP_COMPLETE="data/LIP_dataset/TE440_less_than_1024.txt"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "Starting sequential model evaluations..."
echo "------------------------------------------------"

for model_path in "$MODELS_DIR"/*; do
    # Ensure it is a valid directory
    if [ -d "$model_path" ]; then
        model_name=$(basename "$model_path")
        model_pt="$model_path/bindCORE.pt"
        
        # Skip if the model weights file doesn't exist
        if [ ! -f "$model_pt" ]; then
            echo "[SKIP] No bindCORE.pt found in $model_path"
            echo "------------------------------------------------"
            continue
        fi

        DATASET=""
        H5_FILE=""

        # =====================================================================
        # 1. Map Datasets & H5 files dynamically based on model naming patterns
        # =====================================================================
        
        # ---- STARLING Models (Strictly restricted to < 380 length) ----
        if [[ "$model_name" == *STARLING* ]]; then
            H5_FILE="data/properties/STARLING_derived_properties.h5"
            if [[ "$model_name" == *LIP* ]]; then
                DATASET="data/LIP_dataset/TE440_less_than_380.txt"
            elif [[ "$model_name" == *MoRF* ]]; then
                DATASET="data/MoRF_dataset/test_less_than_380.txt"
            fi

        # ---- IDPFold2 Models ----
        elif [[ "$model_name" == *IDPFold2* ]]; then
            H5_FILE="data/properties/IDPFold2_derived_properties.h5"
            if [[ "$model_name" == *LIP* ]]; then
                DATASET="$LIP_COMPLETE"
            elif [[ "$model_name" == *MoRF* ]]; then
                DATASET="data/MoRF_dataset/test.txt"
            fi

        # ---- AF_CALVADOS Models ----
        elif [[ "$model_name" == *AF_CALVADOS* ]]; then
            H5_FILE="data/properties/single_AF_CALVADOS_derived_properties.h5"
            if [[ "$model_name" == *LIP* ]]; then
                DATASET="$LIP_COMPLETE"
            elif [[ "$model_name" == *MoRF* ]]; then
                DATASET="data/MoRF_dataset/test.txt"
            fi

        # ---- pLM Models ----
        elif [[ "$model_name" == *pLM* ]]; then
            # Note: No pLM h5 was visible in your data/properties list.
            # Left blank here so the --h5 flag is completely omitted. 
            # (If it needs a fallback, e.g. IDPFold2, change to: H5_FILE="path")
            H5_FILE="" 
            
            if [[ "$model_name" == *LIP* ]]; then
                DATASET="$LIP_COMPLETE"
            elif [[ "$model_name" == *MoRF* ]]; then
                DATASET="data/MoRF_dataset/test.txt"
            fi
        fi

        # =====================================================================
        # 2. Build and Execute the Command safely
        # =====================================================================
        if [ -n "$DATASET" ]; then
            echo "[EVALUATING] Model: $model_name"
            echo "Dataset: $DATASET"
            
            # Initialize array with arguments common to all runs
            cmd_args=(--model "$model_pt" --datasets "$DATASET" --output_dir "$OUTPUT_DIR" --device cuda)
            
            # Conditionally append the h5 flag if one was mapped
            if [ -n "$H5_FILE" ]; then
                echo "H5 Properties: $H5_FILE"
                cmd_args+=(--h5 "$H5_FILE")
            else
                echo "H5 Properties: None (Omitted from command)"
            fi
            
            # Run the evaluation. If it crashes, catch it and continue the loop.
            python3 scripts/predict.py "${cmd_args[@]}" || {
                echo "[CRASH DETECTED] Evaluation failed for $model_name! Moving to next..."
            }
            
            echo "------------------------------------------------"
        else
            echo "[SKIP] Could not resolve dataset mapping for cluster: $model_name"
            echo "------------------------------------------------"
        fi
    fi
done

echo "All model evaluations completed!"