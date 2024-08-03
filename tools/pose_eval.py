#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File Name: pose_eval.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/8/3
Last Modified: 2024/8/3
Version: 1.0

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/8/3] wux024: Initial file creation
"""
import argparse
import metrics
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
import yaml
import numpy as np


YOLO_V8 = ['yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 'yolov8l-pose', 'yolov8x-pose']
YOLO_V8_CSPNEXT = ['yolov8n-pose-cspnext', 'yolov8s-pose-cspnext', 'yolov8m-pose-cspnext']


def parse_args():
    parser = argparse.ArgumentParser(description="Validate YOLOv8 pose estimation models on a specified dataset.")
    parser.add_argument('--model', type=str, default='yolov8-pose', help='Path to the dataset directory.')
    parser.add_argument('--dataset', type=str, default='ap10k', help='Name of the dataset.')
    parser.add_argument('--split', type=str, default='test', help='Split of the dataset to evaluate on.')
    return parser.parse_args()  # Return the parsed arguments


if __name__ == '__main__':
    args = parse_args()
    # Load the dataset configuration file
    with open(f'configs/data/{args.dataset}.yaml', 'r') as f:
         dataset_cfg = yaml.load(f, Loader=yaml.FullLoader)
         if 'oks_sigmas' in dataset_cfg.keys():
             oks_sigmas = np.array(dataset_cfg['oks_sigmas'])  # Load the OKS sigmas for pose evaluation
         else:
             oks_sigmas = np.array([1.0 / dataset_cfg['kpt_shape'][0]] * dataset_cfg['kpt_shape'][0])
    # Set the model configuration file
    if args.model == 'yolov8-pose':
        models = YOLO_V8
    elif args.model == 'yolov8-pose-cspnext':
        models = YOLO_V8_CSPNEXT
    elif args.model in YOLO_V8 or args.model in YOLO_V8_CSPNEXT:
        models = [args.model]
    else:
        raise ValueError(f'Invalid model: {args.model}')
    for model in models:
        # Load the COCO annotations and predictions
        if args.split == 'val':
            coco_gt = COCO(f'datasets/{args.dataset}/annotations/val.json')
        elif args.split == 'train':
            coco_gt = COCO(f'datasets/{args.dataset}/annotations/train.json')
        elif args.split == 'test':
            coco_gt = COCO(f'datasets/{args.dataset}/annotations/test.json')
        else:
            raise ValueError(f'Invalid split: {args.split}')
        # COCO evaluation
        coco_dt = coco_gt.loadRes(f'runs/pose/eval/{args.dataset}/{args.dataset}-{model}/predictions.json')
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='keypoints', sigmas=oks_sigmas, use_area=True)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # PCK evaluation
        # EPE evaluation
        # AUC evaluation
