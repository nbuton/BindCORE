#!/bin/bash

source env_BindCORE/bin/activate 
# Define datasets and models
datasets=("LIP" "MoRF")
models=("IDPFold2" "AF_CALVADOS" "STARLING")

# Iterate through all combinations
for ds in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        
        # Define variable for naming
        exp_name="tune_${ds}_${model}"
        
        # Execute command
        python3 scripts/tune_hyperparams.py \
            --search-space "data/models/BindCORE_${ds}_${model}/search_space.yaml" \
            --max-epochs 10 \
            --num-seeds 1 \
            --num-samples 100 \
            --output-dir ./ray_results \
            --exp-name "$exp_name" > "${exp_name}.log" \
            --device cuda \
            
    done
done
