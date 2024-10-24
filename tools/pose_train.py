#!/usr/bin/env python
"""
File Name: pose_train_animalrtpose.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/7/3
Last Modified: 2024/10/24
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
import os
import subprocess
from datetime import datetime

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

def get_pretrained_model(model_yaml):
    """Get the path to the pretrained model if it exists."""
    pretrained_model = f"{model_yaml[:-5]}.pt"
    pretrained_path = os.path.join("./weights", pretrained_model)
    if os.path.exists(pretrained_path):
        return pretrained_path
    else:
        print(f"Warning: Pretrained model {pretrained_model} not found. Skipping...")
        return None

def construct_train_command(args, model_yaml, pretrained_model):
    """Construct the yolo pose train command."""
    datacfg = f"./configs/data/{args.dataset}.yaml"
    modelcfg = f"./configs/models/{args.dataset}/{model_yaml}"
    model_name = build_output_dir(model_yaml, args.seed)
    output_dir = f"./runs/{args.model_type}/{args.dataset}"

    cmd = [
        "yolo",
        "pose",
        "train",
        f"data={datacfg}",
        f"model={modelcfg}",
        f"pretrained={pretrained_model}" if pretrained_model else "",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"project={output_dir}",
        f"name={model_name}",
        f"device={args.device}",
        f"workers={args.workers}",
        f"seed={args.seed}",
        f"cos_lr=True",
        f"resume=True",
        f"pose={args.pose}",
        f"patience={args.patience}",
    ]

    return [arg for arg in cmd if arg]

def main():
    # Get the current date and time for the output directory
    current_date = datetime.now().strftime("%Y%m%d")
    seed_value = int(current_date)

    # Default training settings
    default_settings = {
        "dataset": "ap10k",
        "epochs": 1000,
        "patience": 300,
        "batch": -1,
        "imgsz": 640,
        "device": None,
        "workers": 16,
        "pose": 40.0,
        "seed": seed_value,
        "model_type": "animalrtpose",
        "models": None,
    }

    # Define the argument parser
    parser = argparse.ArgumentParser(
        description="Train pose estimation models on a specified dataset with default or user-provided settings."
    )

    # Required argument
    parser.add_argument("--dataset", type=str, default=default_settings["dataset"], help="Name of the dataset.")

    # Optional arguments
    parser.add_argument("--epochs", type=int, default=default_settings["epochs"], help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=default_settings["patience"], help="Early stopping patience.")
    parser.add_argument("--batch", type=int, default=default_settings["batch"], help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=default_settings["imgsz"], help="Image size.")
    parser.add_argument(
        "--device", type=str, default=default_settings["device"], help="Device to use (e.g., 0, 1, 2, cpu)."
    )
    parser.add_argument("--model_type", type=str, default=default_settings["model_type"], help="Model type.")
    parser.add_argument("--models", type=str, help="Comma-separated list of model codes (n, s, m, l, x).")
    parser.add_argument("--no-pretrained", action="store_true", help="not use a pretrained model.")
    parser.add_argument("--workers", type=int, default=default_settings["workers"], help="Number of workers.")
    parser.add_argument("--seed", type=int, default=default_settings["seed"], help="Random seed.")
    parser.add_argument("--pose", type=float, default=default_settings["pose"], help="Pose loss weight.")

    args = parser.parse_args()

    # Process selected models
    if args.models is None:
        models = parse_models("n,s,m,l,x", model_type=args.model_type)
    else:
        models = parse_models(args.models, model_type=args.model_type)

    # Loop through each model for the given dataset
    for model_yaml in models:
        # Get the path to the pretrained model if it exists
        pretrained_model = get_pretrained_model(model_yaml) if not args.no_pretrained else None

        # Construct the yolo pose train command
        train_cmd = construct_train_command(args, model_yaml, pretrained_model)

        # Run the command
        print(f"Running command: {' '.join(train_cmd)}")
        subprocess.run(train_cmd, check=True)


if __name__ == "__main__":
    main()
