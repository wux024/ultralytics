#!/usr/bin/env bash

# Script: Train YOLOv8 models on a specified dataset with default or user-provided settings.
# Usage: bash tools/val.sh \ 
# --dataset <datase_name> \
# --imgsz <image_size> \
# --batch <batch_size> \
# --save-json \
# --save-hybrid \    # save model in ONNX and TorchScript formats
# --conf <conf_thresh> \
# --iou <iou_thresh> \
# --max-det <max_detections> \
# --half \
# --device <device> \
# --dnn \
# --plots \
# --rect \
# --split <split_name> \
# --models <model_codes>

dataset=ap10k
imgsz=640
batch=16
save_json=False
save_hybrid=False
conf=0.001
iou=0.6
max_det=300
half=False
device=None
dnn=False
plots=False
rect=False
split=test

# Parse optional arguments
while [ "$#" -gt 0 ]; do
    key="$1"
    case $key in
        --dataset)
            dataset="$2"
            shift 2
            ;;
        --imgsz)
            imgsz="$2"
            shift 2
            ;;
        --batch)
            batch="$2"
            shift 2
            ;;
        --save-json)
            save_json=True
            shift
            ;;
        --save-hybrid)
            save_hybrid=True
            shift
            ;;
        --conf)
            conf="$2"
            shift 2
            ;;
        --iou)
            iou="$2"
            shift 2
            ;;
        --max_det)
            max_det="$2"
            shift 2
            ;;
        --half)
            half=True
            shift
            ;;
        --device)
            device="$2"
            shift 2
            ;;
        --dnn)
            dnn=True
            shift
            ;;
        --plots)
            plots=True
            shift
            ;;
        --rect)
            rect=True
            shift 2
            ;;
        --split)
            split="$2"
            shift 2
            ;;
        --models)
            IFS=',' read -ra selected_models <<< "$2"
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
    models=("yolov8n-pose-cspnext.yaml" "yolov8s-pose-cspnext.yaml" "yolov8m-pose-cspnext.yaml" "yolov8l-pose-cspnext.yaml")
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
    
    model_name="${dataset}-${model_yaml%.yaml}"
    model_dir="./runs/pose/train/$dataset/$model_name"
    model=$model_dir/weights/best.pt
    output_dir="./runs/pose/eval/$dataset/$model_name"

    # Launch YOLOv8 pose evaluation command
    echo "Evaluating  $model_yaml on $dataset..."
    yolo pose val \
        model=$model \
        data=./configs/data/"$dataset".yaml \
        imgsz=$imgsz \
        batch=$batch \
        project=$output_dir \
        name=$model_name \
        device=$device \
        conf=$conf \
        iou=$iou \
        max_det=$max_det \
        half=$half \
        save_json=$save_json \
        save_hybrid=$save_hybrid \
        dnn=$dnn \
        plots=$plots \
        rect=$rect \
        split=$split
done