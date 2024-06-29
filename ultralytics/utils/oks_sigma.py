#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File Name: oks_sigma.py
Author: wux024
Email: wux024@nenu.edu.cn
Created On: 2024/6/29
Last Modified: 2024/6/29
Version: 1.0

Overview:
    Provide a concise summary of the file's functionality, objectives, or primary logic implemented.
    
Notes:
    - Modifications should be documented in the "Revision History" section beneath this.
    - Ensure compliance with project coding standards.

Revision History:
    - [2024/6/29] wux024: Initial file creation
"""
import numpy as np
from ultralytics.utils.metrics import OKS_SIGMA


OKS_SIGMA_USER = np.array([1.0 / 17.0]*17)

def GetOKSSigma():
    global OKS_SIGMA_USER
    return OKS_SIGMA_USER

def SetOKSSigma(sigma):
    global OKS_SIGMA_USER
    OKS_SIGMA_USER = np.array(sigma)

def SetDefaultOKSSigma(kpt_nums):
    global OKS_SIGMA_USER
    OKS_SIGMA_USER = np.array([1.0 / kpt_nums]*kpt_nums)