#!/usr/bin/env python
"""
File Name: pose_modify_categories_id.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/7/28
Last Modified: 2024/7/28
Version: 1.0.

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/7/28] wux024: Initial file creation
"""

import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Modify categories id in COCO json file.")
    parser.add_argument("--dataset", type=str, default="ap10k", help="Name of the dataset.")

    args = parser.parse_args()
    return args


def modify_categories_id(json_file):
    with open(json_file) as f:
        data = json.load(f)
    cat_id = set([data["category_id"] for data in data])
    if 0 in cat_id:
        for i in range(len(data)):
            data[i]["category_id"] += 1
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    dataset = args.dataset
    base_dir = os.path.join("runs", "pose", "eval", dataset)

    for dir, subdir, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".json"):
                json_file = os.path.join(dir, file)
                modify_categories_id(json_file)