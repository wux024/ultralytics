#! usr/bin/env python
# -*- coding: utf-8 -*-
"""
File Name: print_model_info.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/7/21
Last Modified: 2024/7/21
Version: 1.0

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/7/21] wux024: Initial file creation
"""


from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLOv8 pose estimation models on a specified dataset.")
    parser.add_argument('--dataset', type=str, default='ap10k', help='Name of the dataset.')
    parser.add_argument('--backbone', type=str, default='yolov8', help='Name of the weights file.')
    return parser.parse_args()  # Return the parsed arguments

if __name__ == '__main__':
    args = parse_args()
    dataset_name = args.dataset
    if args.backbone == 'yolov8':
        model_lists = ['yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 'yolov8l-pose', 'yolov8x-pose']
    elif args.backbone == 'cspnext':
        model_lists = ['yolov8n-pose-cspnext', 'yolov8s-pose-cspnext', 'yolov8m-pose-cspnext', 'yolov8l-pose-cspnext', 'yolov8x-pose-cspnext']
    
    for model_name in model_lists:
        yaml_path = f'configs/models/{dataset_name}/{model_name}.yaml'
        model = YOLO(yaml_path, task='pose')
        model.info()