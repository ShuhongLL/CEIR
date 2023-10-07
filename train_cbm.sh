#!/bin/bash

current_time=$(date +"%H%M")

# Define arrays for the datasets, backbones, and latent dimensions
datasets=("cifar100" "cifar100-20")
backbones=("clip_RN50" "clip_ViT-B/16" "clip_ViT-L/14")
latent_dims=("128" "256")
device="cuda:0"

# Loop through all combinations
for dataset in "${datasets[@]}"; do
    for backbone in "${backbones[@]}"; do
        # Convert the current backbone value, replacing '/' with '-'
        log_backbone=$(echo $backbone | tr '/' '-')
        
        for latent_dim in "${latent_dims[@]}"; do
            # Calculate hidden_dim as twice the latent_dim
            hidden_dim=$((2 * $latent_dim))
            
            # Toggle between cuda:0 and cuda:1
            if [ "$device" == "cuda:0" ]; then
                device="cuda:1"
            else
                device="cuda:0"
            fi
            
            # Construct the command
            cmd="nohup python -u train_cbm.py --dataset $dataset \
            --concept_set data/concept_sets/${dataset}_filtered_gpt4.txt --backbone $backbone \
            --train_vae True --vae_train_set both --vae_hidden_dim $hidden_dim \
            --vae_latent_dim $latent_dim --vae_epochs 450 --save_vae True --device $device \
            > ./log/${dataset}_${log_backbone}_z${latent_dim}_${current_time}.log &"
            
            # Run the command
            echo "Executing: $cmd"
            eval $cmd
        done
    done
done
