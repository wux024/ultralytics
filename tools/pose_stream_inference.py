#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File Name: pose_stream_inference.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/6/19
Last Modified: 2024/10/28
Version: 1.1.

Overview:
    Stream inference script for various pose estimation models on a specified dataset with default or user-provided settings.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/6/19] wux024: Initial file creation
    - [2024/10/28] wux024: Added support for multiple model types and optical field parameters
"""

import argparse
import os
import yaml
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.pose_cfg import SetSkeleton


def parse_models(models_str, model_type="animalrtpose"):
    """Parse the comma-separated list of model codes into a list of model YAML files."""
    
    # Define mappings from model codes to YAML file names for each model type
    model_mappings = {
        "animalrtpose": {
            "n": "animalrtpose-n.yaml",
            "s": "animalrtpose-s.yaml",
            "m": "animalrtpose-m.yaml",
            "l": "animalrtpose-l.yaml",
            "x": "animalrtpose-x.yaml"
        },
        "yolov8": {
            "n": "yolov8n-pose.yaml",
            "s": "yolov8s-pose.yaml",
            "m": "yolov8m-pose.yaml",
            "l": "yolov8l-pose.yaml",
            "x": "yolov8x-pose.yaml"
        },
        "yolo11": {
            "n": "yolo11n-pose.yaml",
            "s": "yolo11s-pose.yaml",
            "m": "yolo11m-pose.yaml",
            "l": "yolo11l-pose.yaml",
            "x": "yolo11x-pose.yaml"
        },
        "spipose": {
            "n": "spipose-n.yaml",
            "s": "spipose-s.yaml",
            "m": "spipose-m.yaml",
            "l": "spipose-l.yaml",
            "x": "spipose-x.yaml"
        },
    }
    
    # Check if the provided model type is valid
    if model_type not in model_mappings:
        raise ValueError(f"Invalid model type: {model_type}. Valid types are {', '.join(model_mappings.keys())}.")
    
    # Get the mapping for the specific model type
    mapping = model_mappings[model_type]
    
    # Split the input string into individual model codes and map them to YAML file names
    models = [mapping[code.strip()] for code in models_str.split(",") if code.strip() in mapping]
    
    return models

def build_paths(model, 
                dataset, 
                model_type, 
                seed, 
                optical_field_sizes=None, 
                sub_optical_field_sizes=None, 
                window_size=None, 
                inverse=False, 
                imgsz_hadamard=None):
    common_base = model
    base_common = f"./runs/{model_type}/train/{dataset}"
    dir_common = f"runs/{model_type}/predict/{dataset}"

    if optical_field_sizes is not None:
        common_base += f"-{optical_field_sizes}x{optical_field_sizes}"
    if sub_optical_field_sizes is not None:
        common_base += f"-sub{sub_optical_field_sizes}x{sub_optical_field_sizes}"
    if window_size is not None:
        common_base += f"-{window_size}x{window_size}"
    if inverse:
        common_base += "-inverse"
    if imgsz_hadamard is not None:
        common_base += f"-imgsz{imgsz_hadamard}"
    if seed is not None:
        common_base += f"-{seed}"
    
    model_path = os.path.join(base_common, common_base, "weights/best.pt")
    save_path = os.path.join(dir_common, common_base)
    
    return model_path, save_path

def save_results(result, save_dir, img, im, im_black, im_white, args):
    """Save the results on different backgrounds."""
    # Update original image size
    result.orig_shape = im.shape[:2]

    # Rescale keypoints
    keypoints_rescaled = result.keypoints.xyn.clone()
    keypoints_rescaled[:, :, 0] *= im.shape[1]
    keypoints_rescaled[:, :, 1] *= im.shape[0]
    keypoints_data_cloned = result.keypoints.data.clone()
    keypoints_data_cloned[:, :, :2] = keypoints_rescaled
    result.keypoints.data = keypoints_data_cloned

    # Save results on different backgrounds
    for bg_img, prefix in [(im, ''), (im_black, 'black_'), (im_white, 'white_')]:
        result.save(
            filename=f"{save_dir}/{prefix}{img}",
            img=bg_img,
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fish", help="dataset to use")
    parser.add_argument("--models", type=str, required=True, help="comma-separated list of model codes")
    parser.add_argument("--model_type", type=str, default="animalrtpose", help="model type")
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
    parser.add_argument("--optical_field_sizes", type=int, default=None, help="optical field sizes for embedding head")
    parser.add_argument("--sub_optical_field_sizes", type=int, default=None, help="sample rate for inference")
    parser.add_argument("--window_size", type=int, nargs='+', default=None, help="window size for embedding head")
    parser.add_argument("--inverse", action="store_true", help="inverse flag for embedding head")
    parser.add_argument("--imgsz_hadamard", type=int, default=None, help="Hadamard image size")
    parser.add_argument("--seed", type=int, default=None, help="seed for inference")
    args = parser.parse_args()

    # Parse the provided models
    models = parse_models(args.models, args.model_type)

    data_cdg = yaml.load(open(f"configs/data/{args.dataset}.yaml"), Loader=yaml.FullLoader)

    for model_yaml in models:
        model_name = os.path.splitext(os.path.basename(model_yaml))[0]
        model_path, save_dir = build_paths(model_name, args.dataset, args.model_type, args.seed, args.optical_field_sizes, args.sub_optical_field_sizes, args.window_size, args.inverse, args.imgsz_hadamard)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model = YOLO(model_path)

        # Set skeleton
        if 'skeleton' in data_cdg.keys():
            SetSkeleton(data_cdg['skeleton'])

        # Determine data paths
        if 'test' in data_cdg.keys():
            data_path = os.path.join("datasets", data_cdg['path'], data_cdg['test'])
            high_data_path = os.path.join("datasets", data_cdg['path'], "images_", "test")
        elif 'val' in data_cdg.keys():
            data_path = os.path.join("datasets", data_cdg['path'], data_cdg['val'])
            high_data_path = os.path.join("datasets", data_cdg['path'], "images_", "val")
        elif 'train' in data_cdg.keys:
            data_path = os.path.join("datasets", data_cdg['path'], data_cdg['train'])
            high_data_path = os.path.join("datasets", data_cdg['path'], "images_", "train")
        else:
            raise ValueError(f"No test or val or train data found in {data_cdg['path']}")

        results = model(
            data_path,
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
            stream=True
        )

        # high_data
        high_datas = os.listdir(high_data_path)

        for result, img in zip(results, high_datas):
            im = cv2.imread(os.path.join(high_data_path, img))
            im_black = np.zeros((im.shape[0], im.shape[1], 3), dtype=np.uint8)
            im_white = np.ones((im.shape[0], im.shape[1], 3), dtype=np.uint8) * 255

            save_results(result, save_dir, img, im, im_black, im_white, args)