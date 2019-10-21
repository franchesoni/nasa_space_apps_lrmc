#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 12:14:02 2018

@author: franchesoni
"""

import numpy as np

def soft_threshold(tau, s):
    return (s > tau) * (s - tau) + (s < -tau) * (s + tau)

def D(tau, X):
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    D = u @ np.diag(soft_threshold(tau, s)) @ vh
    return D

def P_o(X, W):
    return X * W