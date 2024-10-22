#!/usr/bin/env python
"""
File Name: pose_train_cspnext.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/7/3
Last Modified: 2024/7/3
Version: 1.0.

Overview:
    Train YOLOv8 models on a specified dataset with default or user-provided settings.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/7/3] wux024: Initial file creation
"""

import argparse
import os
import subprocess

def get_dataconfig(args):
    if args.spi:
        if args.rate is None:
            raise ValueError("Error: SPI sample rate must be specified for SPI mode.")
        
        if args.spi == "aliasing":
            if args.window:
                return f"{args.dataset}-{args.spi}-{args.rate}-{args.window}"
            else:
                return f"{args.dataset}-{args.spi}-{args.rate}"
        elif args.spi == "sample":
            return f"{args.dataset}-{args.spi}-{args.rate}"
        else:
            raise ValueError(f"Error: Unknown SPI mode '{args.spi}'.")
    else:
        return f"{args.dataset}"

def main():
    # Default training settings
    default_settings = {
        "dataset": "mouse",
        "epochs": 10000,
        "patience": 1000,
        "batch": -1,
        "imgsz": 640,
        "device": None,
        "workers": 8,
        "cos_lr": True,
        "resume": True,
        "seed": 0,
        "pose": 40.0,
    }

    # Define the argument parser
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 models on a specified dataset with default or user-provided settings."
    )

    # Required argument
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset.")

    # Optional arguments
    parser.add_argument("--epochs", type=int, default=default_settings["epochs"], help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=default_settings["patience"], help="Early stopping patience.")
    parser.add_argument("--batch", type=int, default=default_settings["batch"], help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=default_settings["imgsz"], help="Image size.")
    parser.add_argument("--device", type=str, default=default_settings["device"], help="Device to use (e.g., 0, 1, 2, cpu).")
    parser.add_argument("--models", type=str, help="Comma-separated list of model codes (n, s, m, l, x).")
    parser.add_argument("--pretrained", action="store_true", help="Use a pretrained model.")
    parser.add_argument("--workers", type=int, default=default_settings["workers"], help="Number of workers.")
    parser.add_argument("--cos-lr", action="store_true", default=default_settings["cos_lr"], help="Use cosine learning rate schedule.")
    parser.add_argument("--resume", action="store_true", default=default_settings["resume"], help="Resume training.")
    parser.add_argument("--seed", type=int, default=default_settings["seed"], help="Random seed.")
    parser.add_argument("--pose", type=float, default=default_settings["pose"], help="Pose loss weight.")
    parser.add_argument("--spi", type=str, default=None, help="SPI mode")
    parser.add_argument("--rate", type=int, default=None, help="SPI sample rate")
    parser.add_argument("--window", type=str, default=None, help="SPI window size")

    args = parser.parse_args()

    # Set the dataset mode
    data_name = get_dataconfig(args)
    dataconfig = os.path.join("configs", "data", f"{data_name}.yaml")

    # Default models
    model_map = {
        "n": "yolov8n-pose-spipose.yaml",
        "s": "yolov8s-pose-spipose.yaml",
        "m": "yolov8m-pose-spipose.yaml",
        "l": "yolov8l-pose-spipose.yaml",
        "x": "yolov8x-pose-spipose.yaml",
    }

    # Process selected models
    if args.models:
        selected_models = args.models.split(",")
        models = [model_map.get(model_code, None) for model_code in selected_models]
        models = [model for model in models if model is not None]
        if not models:
            raise ValueError(
                "Error: No valid model selected after processing input. Please choose from n, s, m, l, x, or leave empty to train all."
            )
    else:
        models = list(model_map.values())

    # Loop through each model for the given dataset
    for model_yaml in models:
        if args.pretrained:
            pretrained_model = f"{model_yaml[:-5]}.pt"
            pretrained_path = os.path.join("weights", pretrained_model)
            if not os.path.exists(pretrained_path):
                print(f"Pretrained model {pretrained_model} not found. Skipping...")
                pretrained_model = None
            else:
                pretrained_model = pretrained_path
        else:
            pretrained_model = None

        model_name = f"{data_name}-{model_yaml[:-5]}"
        output_dir = os.path.join("runs", "pose", "train", args.dataset)

        # Construct the yolo pose train command
        cmd = [
            "yolo",
            "pose",
            "train",
            f"data={dataconfig}",
            f"model={os.path.join('configs', 'models', args.dataset, model_yaml)}",
            f"pretrained={pretrained_model}",
            f"epochs={args.epochs}",
            f"imgsz={args.imgsz}",
            f"batch={args.batch}",
            f"project={output_dir}",
            f"name={model_name}",
            f"device={args.device}",
            f"cos_lr={args.cos_lr}",
            f"resume={args.resume}",
            f"workers={args.workers}",
            f"seed={args.seed}",
            f"pose={args.pose}",
            f"patience={args.patience}",
        ]

        # Execute the command
        print(f"Training {model_yaml} on {args.dataset}...")
        subprocess.run(cmd)

if __name__ == "__main__":
    main()