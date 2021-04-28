#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 18:30:01 2021

@author: denesh
"""
import numpy as np
import time
import cv2 as cv
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


def plot_comp_efforts(images):
    num_images= len(images)

    ## Vary features
    features = [100,300,500,700,900] #Here edge_thresh = 10
    
    time_orb_feat = np.zeros((5,num_images))
    time_kmeans_feat = np.zeros((5,num_images))
     
    for n in range(len(features)):
      for i in range(num_images):
          img = images[i]  
    
          
    
          # your code here    
          start = time.time()
          # Initiate ORB detector
          orb = cv.ORB_create(nfeatures=features[n],edgeThreshold=10)
          # find the keypoints with ORB
          kp = orb.detect(img,None)
          time_orb_feat[n,i] = (time.time() - start)
    
          feat = []
          for point in kp:
            feat.append(point.pt)
          X = np.array(feat)
          start = time.time()
    
          kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    
          center = kmeans.cluster_centers_
          time_kmeans_feat[n,i] = (time.time() - start)
    
    
    ## Vary edgeThreshold
    edgethresh = [5,10,40,80] #Here nfeautures = 700
    
    time_orb_edge = np.zeros((4,num_images))
    time_kmeans_edge = np.zeros((4,num_images))
     
    for e in range(len(edgethresh)):
      for i in range(num_images):
          img = images[i]  
    
          
    
          # your code here    
          start = time.time()
          # Initiate ORB detector
          orb = cv.ORB_create(nfeatures=700,edgeThreshold=edgethresh[e])
          # find the keypoints with ORB
          kp = orb.detect(img,None)
          time_orb_edge[e,i] = (time.time() - start)
    
          feat = []
          for point in kp:
            feat.append(point.pt)
          X = np.array(feat)
          start = time.time()
    
          kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
    
          center = kmeans.cluster_centers_
          time_kmeans_edge[e,i] = (time.time() - start)
          
          
          
    
    x = features
    y = np.mean(time_orb_feat+time_kmeans_feat, axis=1)  #Total time
    yerr = np.std(time_orb_feat+time_kmeans_feat, axis=1)
    
    fig, ax = plt.subplots()
    
    ax.errorbar(x, y, yerr=yerr,fmt='-o')
    
    ax.set_xlabel('nFeatures')
    ax.set_ylabel('Time (s)')
    ax.set_title('Wallclock time vs nfeatures for all images ')
    plt.show()
    
    x = edgethresh
    y = np.mean(time_orb_edge+time_kmeans_edge, axis=1)  #Total time
    yerr = np.std(time_orb_edge+time_kmeans_edge, axis=1)
    
    fig, ax = plt.subplots()
    
    ax.errorbar(x, y, yerr=yerr,fmt='-o')
    
    ax.set_xlabel('edgeThreshold (pixels)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Wallclock time vs edgeThreshold for all images')
    plt.show()
