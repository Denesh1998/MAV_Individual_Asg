#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 18:34:53 2021

@author: denesh
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def calc_ROC_params(kp,mask):
  #Find number of keypoints that lie on the mask
  TP = 0
  FP = 0
  n = kp.shape[0]
  P = np.sum(mask > 0)#Ground truth of positive class
  N = np.sum(mask == 0)#Ground truth of negative class
 
  mask = mask.reshape(360,360)
  for i in range(n):
    if mask[kp[i,0],kp[i,1]] > 0:
      TP = TP+1
    else:
      FP = FP+1 
  FPR = FP/N
  TPR = TP/P
  return TPR,FPR

def plot_ROC(images,masks):
    images_np = np.stack(images, axis=0 )
    masks_np = np.stack( masks, axis=0 ) #Converting list of arrays to an array
    
    ## Vary features
    features = [100,300,500,700,900] #Here edge_thresh = 10
    num_images = len(images)
    
    TPR_feat = np.zeros((5,num_images))
    FPR_feat = np.zeros((5,num_images))
    
    for n in range(len(features)):
      for i in range(num_images):
          img = images_np[i]  
    
          # Initiate ORB detector
          orb = cv.ORB_create(nfeatures=features[n],edgeThreshold=10)
          # find the keypoints with ORB
          kp = orb.detect(img,None)
    
          feat = []
          for point in kp:
            feat.append(point.pt)
          X = np.array(feat)
          X = X.astype(int) #make sure indices are of type int
          
          TPR_feat[n,i],FPR_feat[n,i] = calc_ROC_params(X,masks_np[i])
    
    ## Vary edgeThreshold
    edgethresh = [5,10,40,80] #Here nfeautures = 700
    
    TPR_edge = np.zeros((4,num_images))
    FPR_edge = np.zeros((4,num_images))
     
    for e in range(len(edgethresh)):
      for i in range(num_images):
          img = images_np[i]  
    
          
    
          # your code here    
          # Initiate ORB detector
          orb = cv.ORB_create(nfeatures=700,edgeThreshold=edgethresh[e])
          # find the keypoints with ORB
          kp = orb.detect(img,None)
    
          feat = []
          for point in kp:
            feat.append(point.pt)
          X = np.array(feat)
          X = X.astype(int) #make sure indices are of type int
    
          TPR_edge[e,i],FPR_edge[e,i] = calc_ROC_params(X,masks_np[i])
    
    
    #ROC curve varying nFeatures
    x = np.mean(FPR_feat, axis=1)
    y = np.mean(TPR_feat, axis=1)  
    
    xerr = np.std(FPR_feat, axis=1)
    yerr = np.std(TPR_feat, axis=1)
    
    fig, ax = plt.subplots()
    
    ax.errorbar(x, y, xerr=xerr,yerr=yerr,fmt='-o')
    
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC curve varying nfeatures for all images ')
    plt.show()
    
    
    #ROC curve varying edgeThreshold
    x = np.mean(FPR_edge, axis=1)
    y = np.mean(TPR_edge, axis=1)  
    
    
    xerr = np.std(FPR_edge, axis=1)
    yerr = np.std(TPR_edge, axis=1)
    
    fig, ax = plt.subplots()
    
    ax.errorbar(x, y, xerr=xerr,yerr=yerr,fmt='-o')
    
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC curve varying edgeThreshold for all images ')
    plt.show()
    