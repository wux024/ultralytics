#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File Name: pose_inference_lowtohigh.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/6/19
Last Modified: 2024/6/19
Version: 1.0

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/6/19] wux024: Initial file creation
"""

import argparse
import os
import yaml
from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.engine.results import Keypoints

YOLO_V8 = ['yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 'yolov8l-pose', 'yolov8x-pose']
YOLO_V8_CSPNEXT = ['yolov8n-pose-cspnext', 'yolov8s-pose-cspnext', 'yolov8m-pose-cspnext', 'yolov8l-pose-cspnext', 'yolov8x-pose-cspnext']


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fish", help="dataset to use")
    parser.add_argument("--model", type=str, default="yolov8-pose", help="model to use")
    parser.add_argument("--conf", type=float, default=0.25, help="object confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IOU threshold for NMS")
    parser.add_argument("--imgsz", type=int, default=640, help="image size")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--max_det", type=int, default=300, help="maximum detections per image")
    parser.add_argument("--vid_stride", type=int, default=1, help="video stride")
    parser.add_argument("--stream_buffer", action="store_true", help="stream buffer")
    parser.add_argument("--visualize", action="store_true", help="visualize detections")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--classes", type=list, default=None, help="filter by class")
    parser.add_argument("--retina_masks", action="store_true", help="use retina mask head")
    parser.add_argument("--embed", action="store_true", help="use embedding head")
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--show_labels", action="store_true", help="show detection class labels")
    parser.add_argument("--show_conf", action="store_true", help="show confidence score")
    parser.add_argument("--show_boxes", action="store_true", help="show boxes")
    parser.add_argument("--show_masks", action="store_true", help="show masks")
    parser.add_argument("--show_probs", action="store_true", help="show probabilities")
    parser.add_argument("--line_width", type=int, default=None, help="line width for boxes")
    parser.add_argument("--kpt_radius", type=int, default=5, help="keypoint radius")
    parser.add_argument("--kpt_line", action="store_true", help="draw keypoint lines")
    parser.add_argument("--optical_field_sizes", type=int, default=64, help="optical field sizes for embedding head")
    parser.add_argument("--sub_optical_field_sizes", type=int, default=64, help="sample rate for inference")
    args = parser.parse_args()
    # Set the model configuration file
    if args.model == 'yolov8-pose':
        models = YOLO_V8
    elif args.model == 'yolov8-pose-cspnext':
        models = YOLO_V8_CSPNEXT
    elif args.model in YOLO_V8 or args.model in YOLO_V8_CSPNEXT:
        models = [args.model]
    else:
        raise ValueError(f'Invalid model: {args.model}')
    
    data_cdg = yaml.load(open(f"configs/data/{args.dataset}.yaml"), Loader=yaml.FullLoader)

    for model in models:
        if args.sub_optical_field_sizes is not None:
            model_path = f"runs/pose/train/{args.dataset}/{args.dataset}-{model}-{args.optical_field_sizes}x{args.optical_field_sizes}-{args.sub_optical_field_sizes}x{args.sub_optical_field_sizes}/weights/best.pt"
        else:
            model_path = f"runs/pose/train/{args.dataset}/{args.dataset}-{model}-{args.optical_field_sizes}x{args.optical_field_sizes}/weights/best.pt"
        if 'test' in data_cdg.keys():
            data_path = f"datasets/{data_cdg['path']}/{data_cdg['test']}"
            high_data_path = f"datasets/{data_cdg['path']}/images_/test"
        elif 'val' in data_cdg.keys():
            data_path = f"datasets/{data_cdg['path']}/{data_cdg['val']}"
            high_data_path = f"datasets/{data_cdg['path']}/images_/val"
        elif 'train' in data_cdg.keys():
            data_path = f"datasets/{data_cdg['path']}/{data_cdg['train']}"
            high_data_path = f"datasets/{data_cdg['path']}/images_/train"
        else:
            raise ValueError(f"No test or val or train data found in {data_cdg['path']}")

        save_dir = f"runs/pose/lowtohigh/{args.dataset}/{args.dataset}-{model}-{args.sample}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model = YOLO(model_path)

        results = model(data_path, 
                        conf=args.conf,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        half=args.half,
                        device=args.device,
                        max_det=args.max_det,
                        vid_stride=args.vid_stride,
                        stream_buffer=args.stream_buffer,
                        visualize=args.visualize,
                        augment=args.augment,
                        agnostic_nms=args.agnostic_nms,
                        classes=args.classes,
                        retina_masks=args.retina_masks,
                        embed=args.embed,
                        stream=True)
        
        # high_data
        high_datas = os.listdir(high_data_path)

        for result, img in zip(results, high_datas):
            im = cv2.imread(high_data_path + '/' + img)
            im_black = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
            im_white = np.ones((im.shape[0], im.shape[1], 3), dtype=np.uint8) * 255
            # Plotting on reconstructed images
            # update original image size
            result.orig_shape = im.shape[:2]
            # updata keypoints
            keypoints_normal = result.keypoints.xyn
            keypoints_rescaled = keypoints_normal.clone()
            keypoints_rescaled[:, :, 0] *= im.shape[1]
            keypoints_rescaled[:, :, 1] *= im.shape[0]
            keypoints_data_cloned = result.keypoints.data.clone()
            keypoints_data_cloned[:, :, :2] = keypoints_rescaled
            result.keypoints.data = keypoints_data_cloned

            result.save(filename = f"{save_dir}/{img}",
                        img = im,
                        conf=args.show_conf,
                        line_width=args.line_width,
                        kpt_radius=args.kpt_radius,
                        kpt_line=args.kpt_line,
                        labels=args.show_labels,
                        boxes=args.show_boxes,
                        masks=args.show_masks,
                        probs=args.show_probs,
                        show=args.show
                        )
            # Plotting on black images
            result.save(filename = f"{save_dir}/black_{img}",
                        img = im_black,
                        conf=args.show_conf,
                        line_width=args.line_width,
                        kpt_radius=args.kpt_radius,
                        kpt_line=args.kpt_line,
                        labels=args.show_labels,
                        boxes=args.show_boxes,
                        masks=args.show_masks,
                        probs=args.show_probs,
                        show=args.show
                        )
            # Plotting on white images
            result.save(filename = f"{save_dir}/white_{img}",
                        img = im_white,
                        conf=args.show_conf,
                        line_width=args.line_width,
                        kpt_radius=args.kpt_radius,
                        kpt_line=args.kpt_line,
                        labels=args.show_labels,
                        boxes=args.show_boxes,
                        masks=args.show_masks,
                        probs=args.show_probs,
                        show=args.show
                        )            