#!/usr/bin/env python
"""
File Name: pose_val_cspnext.py
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

def build_output_dir(model, optical_field_sizes=None, sub_optical_field_sizes=None, window_size=None, seed=None, order=False):
    """Build the save directory based on the provided arguments."""
    base_dir = f"{model}"
    if optical_field_sizes is not None:
        base_dir += f"-{optical_field_sizes}x{optical_field_sizes}"
    if sub_optical_field_sizes is not None:
        base_dir += f"-{sub_optical_field_sizes}x{sub_optical_field_sizes}"
    if window_size is not None:
        base_dir += f"-{window_size[0]}x{window_size[1]}"
    if seed is not None:
        base_dir += f"-{seed}"
    if order:
        base_dir += "-inverse"
    return base_dir

def parse_models(models_str):
    """Parse the comma-separated list of model codes into a list of model YAML files."""
    models = []
    model_codes = models_str.split(",")
    for model_code in model_codes:
        if model_code == "n":
            models.append("spipose-n.yaml")
        elif model_code == "s":
            models.append("spipose-s.yaml")
        elif model_code == "m":
            models.append("spipose-m.yaml")
        elif model_code == "l":
            models.append("spipose-l.yaml")
        elif model_code == "x":
            models.append("spipose-x.yaml")
        else:
            print(f"Warning: Ignoring invalid model code in selection: {model_code}. Valid codes are n, s, m, l, x.")
    return models

def construct_train_command(args, model_yaml):
    """Construct the yolo pose train command."""
    datacfg = f"./configs/data/{args.dataset}.yaml"
    model_name = build_output_dir(
        model_yaml[:-5],
        args.optical_field_sizes,
        args.sub_optical_field_sizes,
        args.window_size,
        args.seed,
        args.order
    )
    model_dir = f"./runs/spipose/train/{args.dataset}"
    output_dir = f"./runs/spipose/eval/{args.dataset}"

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
        "imgsz": 128,
        "batch": 16,
        "conf": 0.001,
        "iou": 0.6,
        "max_det": 300,
        "half": False,
        "device": "0",
        "workers": 16,
        "optical_field_sizes": 128,
        "sub_optical_field_sizes": None,
        "window_size": None,
        "seed": None,
        "order": False,
        "models": None,
    }



    parser = argparse.ArgumentParser(description="Evaluate SPiPose models on a specified dataset.")

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

    # optical_field_sizes selection
    parser.add_argument("--optical_field_sizes", type=int, default=default_settings["optical_field_sizes"], help="Optical field size for training.")

    # sub_optical_field_sizes selection
    parser.add_argument("--sub_optical_field_sizes", type=int, default=default_settings["sub_optical_field_sizes"], help="Sub-optical field size for training.")

    # window_size selection
    parser.add_argument("--window_size", type=int, nargs=2, default=default_settings["window_size"], help="Window size for training.")

    # seed selection
    parser.add_argument("--seed", type=int, default=default_settings["seed"], help="Random seed for training.")

    # order selection
    parser.add_argument("--order", action="store_true", help="Use inverse order for training.")

    # models selection
    parser.add_argument("--models", type=str, default=default_settings["models"], help="Comma-separated list (n, s, m, l, x) of model codes to evaluate.")

    args = parser.parse_args()

    # Default models
    models = [
        "spipose-n.yaml",
        "spipose-s.yaml",
        "spipose-m.yaml",
        "spipose-l.yaml",
        "spipose-x.yaml"
    ]

    # Process selected models
    if args.models is not None:
        models = parse_models(args.models)
    # Check for valid model selection
    if not models:
        raise ValueError(
            "Error: No valid model selected after processing input. Please choose from n, s, m, l, x, or leave empty to evaluate all models."
        )
    

    # Loop through each model for the given dataset
    for model_yaml in models:
        # Construct the yolo pose train command
        cmd = construct_train_command(args, model_yaml)
        # Run the command
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
