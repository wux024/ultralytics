#!/usr/bin/env python

"""
File Name: pose_cfg.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/10/16
Last Modified: 2024/10/16
Version: 1.0.

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.

Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/6/29] wux024: Initial file creation
"""

import numpy as np

OKS_SIGMA_USER = np.array([1.0 / 17.0] * 17)
SKELETON_USER =   [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
  ]


def GetOKSSigma():
    global OKS_SIGMA_USER
    return OKS_SIGMA_USER

def SetOKSSigma(sigma):
    global OKS_SIGMA_USER
    OKS_SIGMA_USER = np.array(sigma)

def SetDefaultOKSSigma(kpt_nums):
    global OKS_SIGMA_USER
    OKS_SIGMA_USER = np.array([1.0 / kpt_nums] * kpt_nums)

def GetSkeleton():
    global SKELETON_USER
    return SKELETON_USER

def SetSkeleton(skeleton):
    global SKELETON_USER
    SKELETON_USER = skeleton
