#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 18:38:48 2021

@author: denesh
"""
import numpy as np
from matplotlib import pyplot as plt

def plot_results():
    features = [100,300,500,700,900] #Here edge_thresh = 10

    x = features
    y = np.array([11,19,20,21,21])
    
    fig, ax = plt.subplots()
    
    ax.plot(x,y)
    ax.set_xlabel('nFeatures')
    ax.set_ylabel('Positive detections')
    ax.set_title('Detections vs nfeatures')
    plt.show()
    edgethresh = [5,10,40,80] #Here nfeautures = 700
    
    x = edgethresh
    y = np.array([19,21,17,9])
    
    fig, ax = plt.subplots()
    
    ax.plot(x,y)
    ax.set_xlabel('edgeThreshold (pixels)')
    ax.set_ylabel('Positive detections')
    ax.set_title('Detections vs edgeThreshold')
    plt.show()
      