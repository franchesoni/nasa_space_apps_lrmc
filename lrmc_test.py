#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 11:38:48 2018

@author: franchesoni
"""
import numpy as np
import matplotlib.pyplot as plt
from lrmc import lrmc
# 10 images to be used, can be changed
files = [
        'yaleB01_P00A+000E+00.pgm',
        'yaleB01_P00A+010E-20.pgm',
        'yaleB01_P00A+020E-10.pgm',
        'yaleB01_P00A+025E+00.pgm',
        'yaleB01_P00A+020E+10.pgm',
        'yaleB01_P00A+015E+20.pgm',
        'yaleB01_P00A+000E+20.pgm',
        'yaleB01_P00A-015E+20.pgm',
        'yaleB01_P00A-020E+10.pgm',
        'yaleB01_P00A-025E+00.pgm'
        ]
# Parameters
N = 10  # number of images
h, w = 192, 168  # height and width 
img_size = 32256  

# Put images in matrix X
X = np.empty((img_size, N))  # D x N data matrix
for i, file in enumerate(files):
    X[:, i] = (plt.imread('YaleB-Dataset/images/yaleB01/'+file)).flatten()

# Create W: D x N binary matrix denoting known (1) or missing (0) entries
removal_percentage = 0
W = np.ones(X.size)
W[0:X.size*removal_percentage//100] = 0
np.random.shuffle(W)
W = W.reshape(X.shape[0], X.shape[1])
#%%
# Run LRMC
tau = 100000000
A, error = lrmc(X, tau, W)
plt.imshow((X*W)[:, 6].reshape(h, w))
plt.figure()
plt.imshow(A[:, 6].reshape(h, w))
plt.figure()
plt.plot(error)

np.save('X_m_0_tau100000000', X*W)
np.save('A_m_0_tau100000000', A)
np.save('error_m_0_tau100000000', error)
