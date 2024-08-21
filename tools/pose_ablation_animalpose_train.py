#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File Name: pose_ablation_animalpose_train.py
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

def main():
    # Default training settings
    default_settings = {
        'epochs': 1000,
        'patience': 300,
        'batch': -1,
        'imgsz': 640,
        'device': None,
        'workers': 8,
        'cos_lr': True,
        'resume': True,
        'seed': 0,
        'pose': 40.0,
        'ablation': 'imgsz',
    }

    # Define the argument parser
    parser = argparse.ArgumentParser(description="Train YOLOv8 models on a specified dataset with default or user-provided settings.")
    
    # Optional arguments
    parser.add_argument('--epochs', type=int, default=default_settings['epochs'], help='Number of training epochs.')
    parser.add_argument('--patience', type=int, default=default_settings['patience'], help='Early stopping patience.')
    parser.add_argument('--batch', type=int, default=default_settings['batch'], help='Batch size.')
    parser.add_argument('--imgsz', type=int, default=default_settings['imgsz'], help='Image size.')
    parser.add_argument('--device', type=str, default=default_settings['device'], help='Device to use (e.g., 0, 1, 2, cpu).')
    parser.add_argument('--workers', type=int, default=default_settings['workers'], help='Number of workers.')
    parser.add_argument('--cos-lr', type=bool, default=default_settings['cos_lr'], help='Use cosine learning rate schedule.')
    parser.add_argument('--resume', type=bool, default=default_settings['resume'], help='Resume training.')
    parser.add_argument('--seed', type=int, default=default_settings['seed'], help='Random seed.')
    parser.add_argument('--pose', type=float, default=default_settings['pose'], help='Pose loss weight.')
    parser.add_argument('--ablation', type=str, default=default_settings['ablation'], help='Ablation study')

    args = parser.parse_args()

    if args.ablation == 'imgsz':
        models = ["yolov8n-pose-cspnext.yaml", "yolov8s-pose-cspnext.yaml", "yolov8m-pose-cspnext.yaml", "yolov8l-pose-cspnext.yaml", "yolov8x-pose-cspnext.yaml"]
    elif args.ablation in ['c2f', 'dw', 'noca', 'nospp', 'nosppca']:
        models = [f"yolov8n-pose-cspnext-{args.ablation}.yaml", f"yolov8s-pose-cspnext-{args.ablation}.yaml", f"yolov8m-pose-cspnext-{args.ablation}.yaml", f"yolov8l-pose-cspnext-{args.ablation}.yaml", f"yolov8x-pose-cspnext-{args.ablation}.yaml"]
    elif args.ablation == 'neck':
        models = ["yolov8n-pose-cspneck.yaml", "yolov8s-pose-cspneck.yaml", "yolov8m-pose-cspneck.yaml", "yolov8l-pose-cspneck.yaml", "yolov8x-pose-cspneck.yaml"]
    else:
        raise ValueError(f"Invalid ablation study: {args.ablation}")


    # Loop through each model for the given dataset
    for model_yaml in models:
        if args.ablation == 'neck':
            pretrained_path = "./weights/" + model_yaml[:12] + ".pt"
        else:
            pretrained_path = "./weights/" + model_yaml[:20] + ".pt"
        
        model_name = f"animalpose-{model_yaml[:-5]}"
        output_dir = f"./runs/pose/train/animalpose"
        if args.ablation == 'imgsz':
            model_name = f"{model_name}-{args.imgsz}"
        # Construct the yolo pose train command
        cmd = [
            "yolo", "pose", "train",
            f"data=./configs/data/animalpose.yaml",
            f"model=./configs/models/animalpose/{model_yaml}",
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
            f"patience={args.patience}",
            f"pretrained={pretrained_path}",
        ]

        # Execute the command
        print(f"Training {model_yaml} on animalpose...")
        subprocess.run(cmd)

if __name__ == '__main__':
    main()