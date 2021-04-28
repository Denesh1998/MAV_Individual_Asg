#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:32:15 2021

@author: denesh
"""

import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

import time
from sklearn.cluster import KMeans
from plot_comp_efforts import *
from plot_ROC import *
from plot_results import *
#Store images as arrays


def get_data(data_dir):
    img_data = []
    mask_data = [] 
    path = data_dir
    for img in os.listdir(path):
         try:
             img_arr = cv.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
             img_arr = cv.resize(img_arr, (360,360), interpolation = cv.INTER_AREA)
             base = os.path.basename(os.path.join(path, img))
             if 'img' in base:
                  img_data.append(img_arr)
             else:
                  gray = cv.cvtColor(img_arr, cv.COLOR_RGB2GRAY)    #Converting mask to grayscale  
                  mask_data.append(gray)
             #resized_arr = cv2.resize(img_arr, (img_size, img_size),interpolation = cv2.INTER_AREA) # Reshaping images to preferred size
             
         except Exception as e:
                print(e)
    return img_data,mask_data


#Function to calculate parameters for ROC curve



#Extract all the images and masks 
images,masks = get_data('WashingtonOBRace/WashingtonOBRace')


#Plot ROC curves
plot_ROC(images,masks)

#Plot the computational efforts plots
plot_comp_efforts(images)

#Plot the results of detections on the 33 images
plot_results()

    