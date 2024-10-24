#!/usr/bin/env python
"""
File Name: pose_modify_categories_id.py
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
import json
import os
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
    parser = argparse.ArgumentParser(description="Modify categories id in COCO json file.")
    parser.add_argument("--dataset", type=str, default="ap10k", help="Name of the dataset.")
    parser.add_argument("--model_type", type=str, default="animalrtpose", help="Type of the model.")
    parser.add_argument("--models", type=str, default="n,s,m,l,x", help="Comma-separated list of model codes.")

    args = parser.parse_args()
    return args


def modify_categories_id(json_file):
    with open(json_file) as f:
        data = json.load(f)
    cat_id = set([data["category_id"] for data in data])
    if 0 in cat_id:
        for i in range(len(data)):
            data[i]["category_id"] += 1
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    base_dir = os.path.join("runs", args.model_type, "eval", dataset)

    for dir, subdir, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                json_file = os.path.join(dir, file)
                modify_categories_id(json_file)
