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

def build_output_dir(model, seed=None):
    """Build the save directory based on the provided arguments."""
    base_dir = f"{model}"
    if seed is not None:
        base_dir += f"-{seed}"
    return base_dir

def parse_models(models_str):
     """Parse the comma-separated list of model codes into a list of model YAML files."""
     models = []
     model_codes = models_str.split(",")
     for model_code in model_codes:
         if model_code == "n":
             models.append("animalrtpose-n.yaml")
         elif model_code == "s":
             models.append("animalrtpose-s.yaml")
         elif model_code == "m":
             models.append("animalrtpose-m.yaml")
         elif model_code == "l":
             models.append("animalrtpose-l.yaml")
         elif model_code == "x":
             models.append("animalrtpose-x.yaml")
         else:
             print(
                 f"Warning: Ignoring invalid model code in selection: {model_code}. Valid codes are n, s, m, l, x."
             )
     return models

def get_pretrained_model(model_yaml):
    """Get the path to the pretrained model if it exists."""
    pretrained_model = f"{model_yaml[:-5]}.pt"
    pretrained_path = os.path.join("./weights", pretrained_model)
    if os.path.exists(pretrained_path):
        return pretrained_path
    else:
        print(f"Warning: Pretrained model {pretrained_model} not found. Skipping...")
        return None

def construct_train_command(args, model_yaml, output_dir, pretrained_model):
    """Construct the yolo pose train command."""
    datacfg = f"./configs/data/{args.dataset}.yaml"
    modelcfg = f"./configs/models/{args.dataset}/{model_yaml}"
    model_name = build_output_dir(model_yaml, args.seed)
    output_dir = f"./runs/animalrtpose/train/{args.dataset}"

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
    }

    # Define the argument parser
    parser = argparse.ArgumentParser(
        description="Train animal real-time pose estimation models on a specified dataset with default or user-provided settings."
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
    parser.add_argument("--models", type=str, help="Comma-separated list of model codes (n, s, m, l, x).")
    parser.add_argument("--no-pretrained", action="store_true", help="not use a pretrained model.")
    parser.add_argument("--workers", type=int, default=default_settings["workers"], help="Number of workers.")
    parser.add_argument("--seed", type=int, default=default_settings["seed"], help="Random seed.")
    parser.add_argument("--pose", type=float, default=default_settings["pose"], help="Pose loss weight.")

    args = parser.parse_args()

    # Default models
    models = [
        "animalrtpose-n.yaml",
        "animalrtpose-s.yaml",
        "animalrtpose-m.yaml",
        "animalrtpose-l.yaml",
        "animalrtpose-x.yaml",
    ]

    # Process selected models
    if args.models:
        models = parse_models(args.models)

    if not models:
        raise ValueError(
            "Error: No valid model selected after processing input. Please choose from n, s, m, l, x, or leave empty to train all."
        )

    # Loop through each model for the given dataset
    for model_yaml in models:
        # Get the path to the pretrained model if it exists
        pretrained_model = get_pretrained_model(model_yaml) if not args.no_pretrained else None

        # Construct the yolo pose train command
        train_cmd = construct_train_command(args, model_yaml, current_date, pretrained_model)

        # Run the command
        print(f"Running command: {' '.join(train_cmd)}")
        subprocess.run(train_cmd)


if __name__ == "__main__":
    main()
