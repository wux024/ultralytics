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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ap10k', help='dataset name')

if __name__ == '__main__':
    # initialize YOLO model
    args = parse_args()

    # set dataset name
    dataset = args.dataset

    model_list = ['yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 'yolov8l-pose', 'yolov8x-pose']
    for model_name in model_list:
        model_name = dataset + '-' + model_name
        model_dir = './runs/pose/train/' + dataset + '/' + model_name + '/weights/best.pt'
        dataset_dir = './configs/data/' + dataset + '.yaml'

        model = YOLO(model_dir)

        metrics = model.val(dataset_dir)

        print(model_name + ' Results:', metrics.pose.map50, metrics.pose.map75, metrics.pose.map)