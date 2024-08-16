#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File Name: pose_stream_inference.py
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

from ultralytics import YOLO

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
    parser.add_argument("--save", action="store_true", help="save results to file")
    parser.add_argument("--show_labels", action="store_true", help="show detection class labels")
    parser.add_argument("--show_conf", action="store_true", help="show confidence score")
    parser.add_argument("--show_boxes", action="store_true", help="show boxes")
    parser.add_argument("--show_masks", action="store_true", help="show masks")
    parser.add_argument("--show_probs", action="store_true", help="show probabilities")
    parser.add_argument("--line_width", type=int, default=None, help="line width for boxes")
    parser.add_argument("--kpt_radius", type=int, default=5, help="keypoint radius")
    parser.add_argument("--kpt_line", action="store_true", help="draw keypoint lines")
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
    
    for model in models:
        model_path = f"runs/pose/train/{args.dataset}/{args.dataset}-{model}/weights/best.pt"
        data_path = f"datasets/{args.dataset}/images/test"

        save_dir = f"runs/pose/predict/{args.dataset}/{args.dataset}-{model}"

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
        
        for i, result in enumerate(results):
            result.save(filename = f"{save_dir}/{i}.jpg",
                        conf=args.show_conf,
                        line_width=args.line_width,
                        kpt_radius=args.kpt_radius,
                        kpt_line=args.kpt_line,
                        labels=args.show_labels,
                        boxes=args.show_boxes,
                        masks=args.show_masks,
                        probs=args.show_probs,
                        show=args.show,
                        )

