#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File Name: pose_val.py
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


from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLOv8 pose estimation models on a specified dataset.")
    parser.add_argument('--dataset', type=str, default='ap10k', help='Name of the dataset.')
    return parser.parse_args()  # Return the parsed arguments

if __name__ == '__main__':
    args = parse_args()  # Actually parse the arguments

    model_list = ['yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 'yolov8l-pose', 'yolov8x-pose']
    
    for model_base_name in model_list:
        model_name = f"{args.dataset}-{model_base_name}"
        model_dir = f'./runs/pose/train/{args.dataset}/{model_name}/weights/best.pt'
        dataset_dir = f'./configs/data/{args.dataset}.yaml'

        try:
            print(f"Validating {model_dir}...")
            model = YOLO(model_dir)
            metrics = model.val(data=dataset_dir)
            
            # Ensure these attributes exist in the returned metrics object before accessing
            if hasattr(metrics.pose, 'map50') and hasattr(metrics.pose, 'map75') and hasattr(metrics.pose, 'map'):
                print(f"{model_name} Results: MAP@0.50={metrics.pose.map50}, MAP@0.75={metrics.pose.map75}, MAP={metrics.pose.map}")
            else:
                print(f"Warning: Some expected metrics were not found for {model_name}.")
        except Exception as e:
            print(f"An error occurred while validating {model_name}: {e}")