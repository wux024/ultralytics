#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: pose_inference.py
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='input.jpg', help='path to input image')
    parser.add_argument('--output', type=str, default='output.jpg', help='path to output image')
    parser.add_argument('--model', type=str, default='yolov8-pose.pt', help='path to YOLO model')

    args = parser.parse_args()

    model = YOLO(args.model)

    results = model(args.input)

    for r in results:
        r.plot(kpt_lines=False,
                labels=False,
                boxes=False,
                masks=False,
                probs=False,
                save=args.output
                )
        