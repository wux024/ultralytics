#!/usr/bin/env python

import argparse
import os
import subprocess
from datetime import datetime


def build_output_dir(model, dataset, optical_field_sizes=None, sub_optical_field_sizes=None, window_size=None, seed=None):
    """Build the save directory based on the provided arguments."""
    base_dir = f"{dataset}"
    if optical_field_sizes is not None:
        base_dir += f"-{optical_field_sizes}x{optical_field_sizes}"
    if sub_optical_field_sizes is not None:
        base_dir += f"-{sub_optical_field_sizes}x{sub_optical_field_sizes}"
    if window_size is not None:
        base_dir += f"-{window_size[0]}x{window_size[1]}"
    if seed is not None:
        base_dir += f"-{seed}"
    base_dir += f"-{model}"
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


def get_pretrained_model_path(model_yaml):
    """Get the path to the pretrained model if it exists."""
    pretrained_model = f"{model_yaml[:-5]}.pt"
    pretrained_path = os.path.join("./weights", pretrained_model)
    if os.path.exists(pretrained_path):
        return pretrained_path
    else:
        print(f"Pretrained model {pretrained_model} not found. Skipping...")
        return None


def construct_train_command(args, model_yaml, pretrained_model):
    """Construct the yolo pose train command."""
    datacfg = f"./configs/data/{args.dataset}.yaml"
    modelcfg = f"./configs/models/{args.dataset}/{model_yaml}"
    model_name = build_output_dir(
        model_yaml[:-5],
        args.dataset,
        args.optical_field_sizes,
        args.sub_optical_field_sizes,
        args.window_size,
        args.seed
    )
    output_dir = f"./runs/spipose/train/{args.dataset}"

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
        f"pose={args.pose}",
        f"patience={args.patience}",
    ]
    return [arg for arg in cmd if arg]  # Filter out any empty strings


def main():
    # Get current date and format it as YYYYMMDD
    current_date = datetime.now().strftime('%Y%m%d')
    seed_value = int(current_date)

    # Default training settings
    default_settings = {
        "epochs": 10000,
        "patience": 5000,
        "batch": -1,
        "imgsz": 128,
        "device": None,
        "workers": 8,
        "pose": 40.0,
        "optical_field_sizes": 128,
        "sub_optical_field_sizes": None,
        "window_size": None,
        "seed": seed_value  # Set the seed value to the current date
    }

    # Define the argument parser
    parser = argparse.ArgumentParser(
        description="Train SPIPose models on a specified dataset with default or user-provided settings."
    )

    # Required argument
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset.")

    # Optional arguments
    parser.add_argument("--epochs", type=int, default=default_settings["epochs"], help="Number of training epochs.")
    parser.add_argument("--patience", type=int, default=default_settings["patience"], help="Early stopping patience.")
    parser.add_argument("--batch", type=int, default=default_settings["batch"], help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=default_settings["imgsz"], help="Image size.")
    parser.add_argument(
        "--device", type=str, default=default_settings["device"], help="Device to use (e.g., 0, 1, 2, cpu)."
    )
    parser.add_argument("--models", type=str, help="Comma-separated list of model codes (n, s, m, l, x).")
    parser.add_argument("--pretrained", action="store_true", help="Use a pretrained model.")
    parser.add_argument("--workers", type=int, default=default_settings["workers"], help="Number of workers.")
    parser.add_argument("--seed", type=int, default=default_settings["seed"], help="Random seed.")
    parser.add_argument("--pose", type=float, default=default_settings["pose"], help="Pose loss weight.")
    parser.add_argument(
        "--optical-field-sizes",
        type=int,
        default=default_settings["optical_field_sizes"],
        help="Optical field size for the entire image.",
    )
    parser.add_argument(
        "--sub-optical-field-sizes",
        type=int,
        default=default_settings["sub_optical_field_sizes"],
        help="Optical field size for sub-regions of the image.",
    )
    parser.add_argument(
        "--window-size",
        nargs=2,
        type=int,
        default=default_settings["window_size"],
        help="Window size for sub-regions of the image.",
    )

    args = parser.parse_args()

    # Default models
    models = [
        "spipose-n.yaml",
        "spipose-s.yaml",
        "spipose-m.yaml",
        "spipose-l.yaml",
        "spipose-x.yaml",
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
        pretrained_model = get_pretrained_model_path(model_yaml) if args.pretrained else None

        cmd = construct_train_command(args, model_yaml, pretrained_model)

        # Execute the command
        print(f"Training {model_yaml} on {args.dataset}...")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()