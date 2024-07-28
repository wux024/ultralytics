#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File Name: pose_modify_categories_id.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/7/28
Last Modified: 2024/7/28
Version: 1.0

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/7/28] wux024: Initial file creation
"""

import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Modify categories id in COCO json file.')
    parser.add_argument('--dataset', type=str, default='ap10k', help='Name of the dataset.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    base_dir = os.path.join('runs', 'pose', 'eval', dataset)

    for dir, subdir, files in os.walk(base_dir):
        print(files)