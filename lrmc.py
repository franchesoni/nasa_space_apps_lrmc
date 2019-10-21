#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 10:12:24 2018

@author: franchesoni
"""
import numpy as np
from aux_functions import D, P_o

def converged(A, X, delta, error):
    error.append(np.linalg.norm(A-X)/np.linalg.norm(X))
    print(error[-1])
    return (error[-1] < delta) or (len(error)>500)

def lrmc(X, tau, W):
    beta = min(2, X.size / np.sum(W))  # min(2, DN/M)
    delta = 0.5e-1
    error = []
    
    Z = A = np.zeros_like(X)
    while not converged(A, X, delta, error):
        A = D(tau, P_o(Z, W))
        Z = Z + beta * (P_o(X, W) - P_o(A, W))
    return A, error