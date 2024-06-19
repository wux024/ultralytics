#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File Name: dataset_modify.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/5/13
Last Modified: 2024/5/13
Version: 1.0

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/5/13] wux024: Initial file creation
"""

import json

datasets = ["pups"]

for dataset in datasets:
    anns = ["train", "val", "test"]
    path = "../datasets/" + dataset + "/annotations/"
    for ann in anns:
        with open(path + ann + ".json") as f:
            data = json.load(f)
        print(data.keys())
        print(len(data["images"]), len(data["annotations"]))
        for num in range(0, len(data["annotations"])):
            # file_name = data['images'][num]['file_name']
            # data['images'][num]['file_name'] = os.path.basename(file_name)
            # width = data['images'][num]['width']
            # height = data['images'][num]['height']
            # full_img_path = '../datasets/' + dataset + '/images/' + ann + '/' + data['images'][num]['file_name']
            #
            # with Image.open(full_img_path) as img:
            #     w_res, h_res = img.size
            #     # modify images
            # data['images'][num]['width'] = w_res
            # data['images'][num]['height'] = h_res

            # modify anns
            keypoints = data["annotations"][num]["keypoints"]
            # data['annotations'][num * 3 + i]['id'] = num * 3 + i
            x = [keypoint for keypoint in keypoints[::3] if keypoint > 0.0]
            y = [keypoint for keypoint in keypoints[1::3] if keypoint > 0.0]
            xmin = min(x) if x else 0.0
            xmax = max(x) if x else 0.0
            ymin = min(y) if y else 0.0
            ymax = max(y) if y else 0.0

            data["annotations"][num]["bbox"] = [xmin, ymin, xmax - xmin, ymax - ymin]
            data["annotations"][num]["area"] = (xmax - xmin) * (ymax - ymin)

        file = open(path + ann + ".json", mode="w")
        json.dump(data, file, indent=4)
        file.close()
