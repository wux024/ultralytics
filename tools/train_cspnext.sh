#!/usr/bin/env bash

# Script: Train YOLOv8 models on a specified dataset with default or user-provided settings.
# Usage: bash tools/train.sh --dataset <dataset_name> 
# [optional: --epochs <num_epochs> --batch <batch_size> --img-size <image_size> --device <device> --models <model_list>] 

# Default training settings
dataset=ap10k
epochs=1000
patience=100
batch=-1
imgsz=640
device=None
workers=8
cos_lr=True
resume=True
pretrained=True
seed=0
pose=12.0


# Ensure a dataset name is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 --dataset <dataset_name> 
    [optional: --epochs <num_epochs> 
    --patience <early_stopping_patience> 
    --batch <batch_size> 
    --imgsz <image_size> 
    --device <device> 
    --workers <num_workers> 
    --cos-lr <cosine_lr_schedule> 
    --resume <resume_training> 
    --pretrained <use_pretrained_model> 
    --seed <random_seed> 
    --models <model_list>]"
    exit 1
fi

# Parse optional arguments
while [[ $# -gt 1 ]]; do
    key="$1"
    case $key in
        --dataset)
            dataset="$2"
            shift 2
            ;;
        --epochs)
            epochs="$2"
            shift 2
            ;;
        --patience)
            patience="$2"
            shift 2
            ;;
        --batch)
            batch="$2"
            shift 2
            ;;
        --imgsz)
            imgsz="$2"
            shift 2
            ;;
        --device)
            device="$2"
            shift 2
            ;;
        --models)
            IFS=',' read -ra selected_models <<< "$2"
            shift 2
            ;;
        --pretrained)
            pretrained="$2"
            shift 2
            ;;
        --workers)
            workers="$2"
            shift 2
            ;;
        --cos-lr)
            cos_lr="$2"
            shift 2
            ;;
        --resume)
            resume="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --pose)
            pose="$2"
            shift 2
            ;;
        *)  # unknown option
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set models based on selection
models=()
if [ -z "$selected_models" ]; then
    # If no specific models are selected, train all by default
    models=("yolov8n-pose-cspnext.yaml" "yolov8s-pose-cspnext.yaml" "yolov8m-pose-cspnext.yaml" "yolov8l-pose-cspnext.yaml" "yolov8x-pose-cspnext.yaml")
else
    # Process selected models
    for model_code in "${selected_models[@]}"; do
        case $model_code in
            n)
                models+=("yolov8n-pose-cspnext.yaml")
                ;;
            s)
                models+=("yolov8s-pose-cspnext.yaml")
                ;;
            m)
                models+=("yolov8m-pose-cspnext.yaml")
                ;;
            l)
                models+=("yolov8l-pose-cspnext.yaml")
                ;;
            x)
                models+=("yolov8x-pose-cspnext.yaml")
                ;;
            *)
                echo "Warning: Ignoring invalid model code in selection: $model_code. Valid codes are n, s, m, l, x."
                ;;
        esac
    done
fi

# Check if any valid models have been selected
if [ ${#models[@]} -eq 0 ]; then
    echo "Error: No valid model selected after processing input. Please choose from n, s, m, l, x, or leave empty to train all."
    exit 1
fi

# Loop through each model for the given dataset
for model_yaml in "${models[@]}"; do
    if [ $pretrained = True ]; then
        # If pretrained is set, use the default pretrained model for the dataset
        pretrained_model="${model_yaml%.yaml}.pt"
        # Ensure the pretrained model exists before attempting to train
        if [ ! -f "./weights/$pretrained_model" ]; then
            echo "Pretrained model $pretrained_model not found. Skipping..."
            pretrained_model=False
            continue
        fi
    else
        # If pretrained is not set, use the default model for the dataset
        pretrained_model=False
    fi
    
    model_name="${dataset}-${model_yaml%.yaml}"
    output_dir="./runs/pose/train/$dataset"

    # Launch YOLOv8 pose training command
    echo "Training $model_yaml on $dataset..."
    yolo pose train \
        data=./configs/data/"$dataset".yaml \
        model=./configs/models/"$dataset"/"$model_yaml" \
        pretrained=$pretrained_model \
        epochs=$epochs \
        imgsz=$imgsz \
        batch=$batch \
        project=$output_dir \
        name=$model_name \
        device=$device \
        cos_lr=$cos_lr \
        resume=$resume \
        workers=$workers \
        seed=$seed \
        pose=$pose \
        patience=$patience
done