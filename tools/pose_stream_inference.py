#!/usr/bin/env python
"""
File Name: pose_stream_inference.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/6/19
Last Modified: 2024/6/19
Version: 1.0.

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/6/19] wux024: Initial file creation
"""

import argparse
import os
import yaml

from ultralytics import YOLO
from ultralytics.utils.pose_cfg import SetSkeleton

ANIMALRTPOSE = [
    "animalpose-n",
    "animalpose-s",
    "animalpose-m",
    "animalpose-l",
    "animalpose-x",
]

def build_model_path(model, dataset, seed):
    base_path = f"./runs/animalrtpose/train/{dataset}/{model}"
    if seed is not None:
        base_path += f"-{seed}"
    base_path = os.path.join(base_path, "weights/best.pt")
    return base_path

def build_save_dir(dataset, model, seed):
    save_dir = f"runs/animalrtpose/predict/{dataset}/{model}"
    if seed is not None:
        save_dir += f"-{seed}"
    return save_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fish", help="dataset to use")
    parser.add_argument("--model", type=str, default="animalrtpose-n", help="model to use")
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
    parser.add_argument("--seed", type=int, default=None, help="seed for inference")
    args = parser.parse_args()


    data_cdg = yaml.load(open(f"configs/data/{args.dataset}.yaml"), Loader=yaml.FullLoader)
    
    if "skeleton" in data_cdg.keys():
        SetSkeleton(data_cdg["skeleton"])
    
    if args.model in ANIMALRTPOSE:
        models = [args.model]
    elif args.model == "all":
        models = ANIMALRTPOSE
    else:
        raise ValueError(f"Invalid model: {args.model}")
    

    for model in models:
        model_path = build_model_path(model, args.dataset, args.seed)
        save_dir = build_save_dir(args.dataset, model, args.seed)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model = YOLO(model_path)

        if "test" in data_cdg.keys():
            data_path = f"datasets/{data_cdg['path']}/{data_cdg['test']}"
        elif "val" in data_cdg.keys():
            data_path = f"datasets/{data_cdg['path']}/{data_cdg['val']}"
        elif "train" in data_cdg.keys():
            data_path = f"datasets/{data_cdg['path']}/{data_cdg['train']}"
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
            stream=True,
        )

        for i, result in enumerate(results):
            img_name = os.path.basename(result.path)
            result.save(
                filename=f"{save_dir}/{img_name}",
                conf=args.show_conf,
                line_width=args.line_width,
                kpt_radius=args.kpt_radius,
                kpt_line=args.kpt_line,
                labels=args.show_labels,
                boxes=args.show_boxes,
                masks=args.show_masks,
                probs=args.show_probs,
                show=args.show,
            )
