#!/usr/bin/env python
"""
File Name: pose_val.py
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 models on a specified dataset.")

    # Add optional arguments
    parser.add_argument("--dataset", type=str, default="ap10k", help="Name of the dataset.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--save-json", action="store_true", help="Save results in JSON format.")
    parser.add_argument("--save-hybrid", action="store_true", help="Save model in ONNX and TorchScript formats.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.6, help="IOU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum number of detections.")
    parser.add_argument("--half", action="store_true", help="Use half precision.")
    parser.add_argument("--device", type=str, help="Device to use (e.g., 0, 1, 2, cpu).")
    parser.add_argument("--dnn", action="store_true", help="Use DNN backend.")
    parser.add_argument("--plots", action="store_true", help="Generate plots.")
    parser.add_argument("--rect", action="store_true", help="Use rectangular training.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split name.")
    parser.add_argument("--models", type=str, help="Comma-separated list of model codes (n, s, m, l, x).")

    args = parser.parse_args()

    # Default models
    models = ["yolov8n-pose.yaml", "yolov8s-pose.yaml", "yolov8m-pose.yaml", "yolov8l-pose.yaml", "yolov8x-pose.yaml"]

    # Process selected models
    if args.models:
        selected_models = args.models.split(",")
        models = []
        for model_code in selected_models:
            if model_code == "n":
                models.append("yolov8n-pose.yaml")
            elif model_code == "s":
                models.append("yolov8s-pose.yaml")
            elif model_code == "m":
                models.append("yolov8m-pose.yaml")
            elif model_code == "l":
                models.append("yolov8l-pose.yaml")
            elif model_code == "x":
                models.append("yolov8x-pose.yaml")
            else:
                print(
                    f"Warning: Ignoring invalid model code in selection: {model_code}. Valid codes are n, s, m, l, x."
                )

    if not models:
        raise ValueError(
            "Error: No valid model selected after processing input. Please choose from n, s, m, l, x, or leave empty to train all."
        )

    # Loop through each model for the given dataset
    for model_yaml in models:
        model_name = f"{args.dataset}-{model_yaml[:-5]}"
        model_dir = f"./runs/pose/train/{args.dataset}/{model_name}"
        model = f"{model_dir}/weights/best.pt"
        output_dir = f"./runs/pose/eval/{args.dataset}"

        # Construct the yolo pose val command
        cmd = [
            "yolo",
            "pose",
            "val",
            f"model={model}",
            f"data=./configs/data/{args.dataset}.yaml",
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

        # Execute the command
        print(f"Evaluating {model_yaml} on {args.dataset}...")
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
