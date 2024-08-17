#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File Name: pose_normal_inference.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/6/19
Last Modified: 2024/6/19
Version: 1.0

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/6/19] wux024: Initial file creation
"""

import argparse

from ultralytics import YOLO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="path/to/video.mp4", help="dataset to use")
    parser.add_argument("--data", type=str, default=None, help="path to dataset")
    parser.add_argument("--stream", action="store_true", help="stream")
    parser.add_argument("--model", type=str, default="path/to/best.pt", help="model to use")
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
    parser.add_argument("--save", action="store_true", help="save results to file")
    parser.add_argument("--save_frames", action="store_true", help="save frames with detections")
    parser.add_argument("--save_txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save_conf", action="store_true", help="save confidences in")
    parser.add_argument("--save_crop", action="store_true", help="save cropped detections")
    parser.add_argument("--show_labels", action="store_true", help="show detection class labels")
    parser.add_argument("--show_conf", action="store_true", help="show confidence score")
    parser.add_argument("--show_boxes", action="store_true", help="show boxes")
    parser.add_argument("--line_width", type=int, default=None, help="line width for boxes")
    parser.add_argument("--kpt_line", action="store_true", help="draw keypoint lines")
    parser.add_argument("--kpt_radius", type=int, default=5, help="keypoint radius")
    parser.add_argument("--project", default="runs/video_results", help="save results to project/name")
    args = parser.parse_args()


    model = YOLO(args.model)

    if args.data is not None:
        data = f"configs/data/{args.data}.yaml"

    model.predict(source=args.source,
                  data=data,
                  stream=args.stream,
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
                  show=args.show,
                  save=args.save,
                  save_frames=args.save_frames,
                  save_txt=args.save_txt,
                  save_conf=args.save_conf,
                  save_crop=args.save_crop,
                  show_labels=args.show_labels,
                  show_conf=args.show_conf,
                  show_boxes=args.show_boxes,
                  line_width=args.line_width,
                  kpt_line=args.kpt_line,
                  kpt_radius=args.kpt_radius,
                  project=args.project
                  )

