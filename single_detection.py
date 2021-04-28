#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 18:23:45 2021

@author: denesh
"""


import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

import time
from sklearn.cluster import KMeans


from matplotlib import pyplot as plt
img = cv.imread('WashingtonOBRace/WashingtonOBRace/img_404.png') [...,::-1]
# Initiate ORB detector

#Create orb object
orb = cv.ORB_create(nfeatures=700,edgeThreshold=10)
# find the keypoints with ORB
kp = orb.detect(img,None)

img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()
feat = []
for point in kp:
  feat.append(point.pt)
X = np.array(feat)
start = time.time()

kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

center = kmeans.cluster_centers_

center =  center. astype(int)
#img[center] = 256
x = center[:,0]
y = center[:,1]

# Here is where you can obtain the coordinate you are looking for
#img = cv.imread('gdrive/MyDrive/data/Gates/train/image/8.png')[...,::-1]

fig, ax = plt.subplots()
ax.imshow(img)
ax.scatter(x, y,c = 'red')
plt.show()