#!/usr/bin/env python
"""
File Name: pose_coco_eval.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/7/28
Last Modified: 2024/7/28
Version: 1.0.

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/7/28] wux024: Initial file creation
"""

import argparse

import numpy as np
import yaml
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

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
        }
    }
    
    # Check if the provided model type is valid
    if model_type not in model_mappings:
        raise ValueError(f"Invalid model type: {model_type}. Valid types are {', '.join(model_mappings.keys())}.")
    
    # Get the mapping for the specific model type
    mapping = model_mappings[model_type]
    
    # Split the input string into individual model codes and map them to YAML file names
    models = [mapping[code.strip()] for code in models_str.split(",") if code.strip() in mapping]
    
    return models

def parse_args():
    parser = argparse.ArgumentParser(description="Validate pose estimation models on a specified dataset.")
    parser.add_argument("--model-type", type=str, default="animalrtpose", help="Path to the dataset directory.")
    parser.add_argument("--models", type=str, default="n", help="Comma-separated list of model codes to evaluate.")
    parser.add_argument("--dataset", type=str, default="ap10k", help="Name of the dataset.")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset to evaluate on.")
    return parser.parse_args()  # Return the parsed arguments


def parse_args():
    parser = argparse.ArgumentParser(description="Validate pose estimation models on a specified dataset.")
    parser.add_argument("--model-type", type=str, default="animalrtpose", help="Type of the model (e.g., animalrtpose, yolov8, yolo11).")
    parser.add_argument("--models", type=str, default="n", help="Comma-separated list of model codes to evaluate (e.g., n,s,m).")
    parser.add_argument("--dataset", type=str, default="ap10k", help="Name of the dataset.")
    parser.add_argument("--split", type=str, default="test", help="Split of the dataset to evaluate on (e.g., train, val, test).")
    return parser.parse_args()

def load_dataset_config(dataset_name):
    """Load the dataset configuration file."""
    with open(f"configs/data/{dataset_name}.yaml") as f:
        dataset_cfg = yaml.load(f, Loader=yaml.FullLoader)
        oks_sigmas = np.array(dataset_cfg.get("oks_sigmas", [1.0 / dataset_cfg["kpt_shape"][0]] * dataset_cfg["kpt_shape"][0]))
    return oks_sigmas

def load_coco_annotations(dataset_name, split):
    """Load the COCO annotations file."""
    return COCO(f"datasets/{dataset_name}/annotations/{split}.json")

def main():
    args = parse_args()
    oks_sigmas = load_dataset_config(args.dataset)
    models = parse_models(args.models, model_type=args.model_type)
    
    for model in models:
        coco_gt = load_coco_annotations(args.dataset, args.split)
        coco_dt = coco_gt.loadRes(f"runs/{args.model_type}/eval/{args.dataset}/{model}/predictions.json")
        
        # Evaluate the predictions
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="keypoints", sigmas=oks_sigmas, use_area=True)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

if __name__ == "__main__":
    main()
