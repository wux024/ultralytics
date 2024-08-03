#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File Name: pose_tune.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/7/28
Last Modified: 2024/7/28
Version: 1.0

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/7/28] wux024: Initial file creation
"""

from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLOv8 pose estimation models on a specified dataset.")
    parser.add_argument('--model', type=str, default='yolov8n-pose', help='Path to the dataset directory.')
    parser.add_argument('--dataset', type=str, default='ap10k', help='Name of the dataset.')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to run the validation.')
    parser.add_argument('--iterations', type=int, default=300, help='Number of iterations to run the validation.')
    parser.add_argument('--device', type=str, default='0,1', help='Device to run the validation on.')
    parser.add_argument('--batch', type=int, default=256, help='Batch size for the validation.')
    return parser.parse_args()  # Return the parsed arguments


if __name__ == '__main__':
    args = parse_args()  # Actually parse the arguments

    model_name = f"{args.dataset}-{args.model}"
    model_dir = f'./runs/pose/train/{args.dataset}/{model_name}/weights/best.pt'
    dataset_dir = f'./configs/data/{args.dataset}.yaml'
    config_dir = f'./configs/pose/train/{args.dataset}/{model_name}/args.yaml'

    device = args.device
    epochs = args.epochs
    iterations = args.iterations
    batch = args.batch
    project_dir = f'./runs/pose/tunings/{args.dataset}'
    try:
        print(f"Validating {model_dir}...")
        model = YOLO(model_dir)
        metrics = model.tune(data=dataset_dir, 
                             epochs=epochs, 
                             iterations=iterations,
                             device=device, 
                             batch=batch, 
                             optimizer='AdamW', 
                             project=project_dir, 
                             name=model_name, 
                             )
    except Exception as e:
        print(f"An error occurred while validating {model_name}: {e}")