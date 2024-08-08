#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File Name: pose_train.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/7/3
Last Modified: 2024/7/3
Version: 1.0

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/7/3] wux024: Initial file creation
"""

import argparse
import subprocess
import os

def main():
    # Default training settings
    default_settings = {
        'dataset': 'ap10k',
        'epochs': 1000,
        'patience': 100,
        'batch': -1,
        'imgsz': 640,
        'device': None,
        'workers': 8,
        'cos_lr': True,
        'resume': True,
        'pretrained': True,
        'seed': 0,
        'pose': 12.0,
    }

    # Define the argument parser
    parser = argparse.ArgumentParser(description="Train YOLOv8 models on a specified dataset with default or user-provided settings.")
    
    # Required argument
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset.')
    
    # Optional arguments
    parser.add_argument('--epochs', type=int, default=default_settings['epochs'], help='Number of training epochs.')
    parser.add_argument('--patience', type=int, default=default_settings['patience'], help='Early stopping patience.')
    parser.add_argument('--batch', type=int, default=default_settings['batch'], help='Batch size.')
    parser.add_argument('--imgsz', type=int, default=default_settings['imgsz'], help='Image size.')
    parser.add_argument('--device', type=str, default=default_settings['device'], help='Device to use (e.g., 0, 1, 2, cpu).')
    parser.add_argument('--models', type=str, help='Comma-separated list of model codes (n, s, m, l, x).')
    parser.add_argument('--pretrained', type=bool, default=default_settings['pretrained'], help='Use pretrained models.')
    parser.add_argument('--workers', type=int, default=default_settings['workers'], help='Number of workers.')
    parser.add_argument('--cos-lr', type=bool, default=default_settings['cos_lr'], help='Use cosine learning rate schedule.')
    parser.add_argument('--resume', type=bool, default=default_settings['resume'], help='Resume training.')
    parser.add_argument('--seed', type=int, default=default_settings['seed'], help='Random seed.')
    parser.add_argument('--pose', type=float, default=default_settings['pose'], help='Pose loss weight.')

    args = parser.parse_args()

    # Default models
    models = ["yolov8n-pose.yaml", "yolov8s-pose.yaml", "yolov8m-pose.yaml", "yolov8l-pose.yaml", "yolov8x-pose.yaml"]

    # Process selected models
    if args.models:
        selected_models = args.models.split(',')
        models = []
        for model_code in selected_models:
            if model_code == 'n':
                models.append("yolov8n-pose.yaml")
            elif model_code == 's':
                models.append("yolov8s-pose.yaml")
            elif model_code == 'm':
                models.append("yolov8m-pose.yaml")
            elif model_code == 'l':
                models.append("yolov8l-pose.yaml")
            elif model_code == 'x':
                models.append("yolov8x-pose.yaml")
            else:
                print(f"Warning: Ignoring invalid model code in selection: {model_code}. Valid codes are n, s, m, l, x.")

    if not models:
        raise ValueError("Error: No valid model selected after processing input. Please choose from n, s, m, l, x, or leave empty to train all.")

    # Loop through each model for the given dataset
    for model_yaml in models:
        if args.pretrained:
            # If pretrained is set, use the default pretrained model for the dataset
            pretrained_model = f"{model_yaml[:-5]}.pt"
            # Ensure the pretrained model exists before attempting to train
            pretrained_path = os.path.join("./weights", pretrained_model)
            if not os.path.exists(pretrained_path):
                print(f"Pretrained model {pretrained_model} not found. Skipping...")
                pretrained_model = False
            else:
                pretrained_model = pretrained_path
        else:
            # If pretrained is not set, use the default model for the dataset
            pretrained_model = False

        model_name = f"{args.dataset}-{model_yaml[:-5]}"
        output_dir = f"./runs/pose/train/{args.dataset}"

        # Construct the yolo pose train command
        cmd = [
            "yolo", "pose", "train",
            f"data=./configs/data/{args.dataset}.yaml",
            f"model=./configs/models/{args.dataset}/{model_yaml}",
            f"pretrained={pretrained_model}",
            f"epochs={args.epochs}",
            f"imgsz={args.imgsz}",
            f"batch={args.batch}",
            f"project={output_dir}",
            f"name={model_name}",
            f"device={args.device}",
            f"cos_lr={args.cos_lr}",
            f"resume={args.resume}",
            f"workers={args.workers}",
            f"seed={args.seed}",
            f"pose={args.pose}",
            f"patience={args.patience}"
        ]

        # Execute the command
        print(f"Training {model_yaml} on {args.dataset}...")
        subprocess.run(cmd)

if __name__ == '__main__':
    main()