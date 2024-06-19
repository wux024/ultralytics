#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File Name: dataset_verify.py
Author: wux024
Email:   <!-- Add this line to include the email address -->
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

from PIL import Image
from pycocotools.coco import COCO

datasets = ["web", "wild"]

for dataset in datasets:
    anns = ["train", "val", "test"]
    path = "../datasets/lote/" + dataset + "/annotations/"
    for ann in anns:
        coco = COCO(path + ann + ".json")
        count = 0
        img_ids = coco.getImgIds()
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = img_info["file_name"]
            w, h = img_info["width"], img_info["height"]
            full_img_path = "../datasets/lote/" + dataset + "/images/" + ann + "/" + img_path
            with Image.open(full_img_path) as img:
                w_, h_ = img.size
            if w_ != w or h_ != h:
                count += 1
        print(dataset + ":" + ann, len(img_ids), count)
