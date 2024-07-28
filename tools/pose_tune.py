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
    return parser.parse_args()  # Return the parsed arguments


if __name__ == '__main__':
    args = parse_args()  # Actually parse the arguments

    model_name = f"{args.dataset}-{args.model}"
    model_dir = f'./runs/pose/train/{args.dataset}/{model_name}/weights/best.pt'
    dataset_dir = f'./configs/data/{args.dataset}.yaml'


    try:
        print(f"Validating {model_dir}...")
        model = YOLO(model_dir)
        metrics = model.tune(data=dataset_dir, 
                             epochs=100, 
                             iterations=300)
    except Exception as e:
        print(f"An error occurred while validating {model_name}: {e}")