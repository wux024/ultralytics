#!/usr/bin/env python
"""
File Name: pose_val_animalrtpose.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/7/3
Last Modified: 2024/7/3
Version: 1.0.

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/7/3] wux024: Initial file creation
"""

import argparse
import subprocess
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

def build_output_dir(model, seed=None):
    """Build the save directory based on the provided arguments."""
    base_dir = f"{model}"
    if seed is not None:
        base_dir += f"-{seed}"
    return base_dir


def construct_val_command(args, model_yaml):
    """Construct the yolo pose train command."""
    datacfg = f"./configs/data/{args.dataset}.yaml"
    model_name = build_output_dir(model_yaml[:-5], args.seed)
    model_dir = f"./runs/{args.model_type}/train/{args.dataset}"
    output_dir = f"./runs/{args.model_type}/eval/{args.dataset}"
    model = os.path.join(model_dir, model_name, "weights/best.pt")

            # Construct the yolo pose val command
    cmd = [
            "yolo",
            "pose",
            "val",
            f"data={datacfg}",
            f"model={model}",
            f"imgsz={args.imgsz}",
            f"batch={args.batch}",
            f"project={output_dir}",
            f"name={model_name}",
            f"device={args.device}",
            f"conf={args.conf}",
            f"iou={args.iou}",
            f"max_det={args.max_det}",
            f"half={args.half}",
            f"save_json={args.save_json}",
            f"save_hybrid={args.save_hybrid}",
            f"dnn={args.dnn}",
            f"plots={args.plots}",
            f"rect={args.rect}",
            f"split={args.split}",
        ]
    
    return [arg for arg in cmd if arg]  # Filter out any empty strings


def main():

    # Default training settings
    default_settings = {
        "dataset": "mouse",
        "imgsz": 640,
        "batch": 16,
        "conf": 0.001,
        "iou": 0.6,
        "max_det": 300,
        "half": False,
        "device": "0",
        "workers": 16,
        "seed": None,
        "models": None,
        "model_type": "animalrtpose",
    }

    parser = argparse.ArgumentParser(description="Evaluate animalrtpose models on a specified dataset.")

    # Dataset selection
    parser.add_argument("--dataset", type=str, default=default_settings["dataset"], help="Dataset to evaluate on.")

    # imgsz selection
    parser.add_argument("--imgsz", type=int, default=default_settings["imgsz"], help="Image size to use for training.")

    # batch selection
    parser.add_argument("--batch", type=int, default=default_settings["batch"], help="Batch size for training.")

    # save_json selection
    parser.add_argument("--save_json", action="store_true", help="Save JSON detections.")

    # save_hybrid selection
    parser.add_argument("--save_hybrid", action="store_true", help="Save hybrid detections.")

    # conf selection
    parser.add_argument("--conf", type=float, default=default_settings["conf"], help="Confidence threshold for detections.")

    # iou selection
    parser.add_argument("--iou", type=float, default=default_settings["iou"], help="IoU threshold for NMS.")

    # max_det selection
    parser.add_argument("--max_det", type=int, default=default_settings["max_det"], help="Maximum number of detections per image.")

    # half selection
    parser.add_argument("--half", action="store_true", help="Use half precision for inference.")

    # device selection
    parser.add_argument("--device", type=str, default=default_settings["device"], help="Device to use for inference.")

    # workers selection
    parser.add_argument("--workers", type=int, default=default_settings["workers"], help="Number of workers for data loading.")

    # seed selection
    parser.add_argument("--seed", type=int, default=default_settings["seed"], help="Random seed for training.")

    # order selection
    parser.add_argument("--order", action="store_true", help="Use inverse order for training.")

    # models selection
    parser.add_argument("--models", type=str, default=default_settings["models"], help="Comma-separated list (n, s, m, l, x) of model codes to evaluate.")

    # models type selection
    parser.add_argument("--model-type", type=str, default="animalrtpose", help="Model type to evaluate. Currently only animalrtpose is supported.")

    args = parser.parse_args()

    # Process selected models
    if args.models is None:
        models = parse_models("n,s,m,l,x", model_type=args.model_type)
    else:
        models = parse_models(args.models, model_type=args.model_type)
    

    # Loop through each model for the given dataset
    for model_yaml in models:
        # Construct the yolo pose train command
        val_cmd = construct_val_command(args, model_yaml)
        # Run the command
        subprocess.run(val_cmd, check=True)


if __name__ == "__main__":
    main()
