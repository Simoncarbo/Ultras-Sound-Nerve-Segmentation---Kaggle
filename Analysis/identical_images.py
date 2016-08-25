# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 10:59:29 2016

@author: sicarbonnell
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2

from Utils.import_data import load_train_data


##import and pre-process data
#train_imgs, train_masks, train_index = load_train_data()
#train_masks = train_masks/255 # convert to 0,1
#train_imgs = train_imgs.astype(np.float)

img1 = np.where((train_index[:,0] == 1) & (train_index[:,1] ==9))
img2 = np.where((train_index[:,0] == 1) & (train_index[:,1] ==14))
#print(np.max(train_imgs[img1]-train_imgs[img2]))

diff = train_imgs[img1,0]-train_imgs[img2,0]
diff = diff[0,0]
#cv2.imshow('hello2',diff.astype(np.uint8))
#cv2.imshow('hello2',train_imgs[1,0].astype(np.uint8))

R,G,B = train_imgs[img1,0]/2,train_imgs[img1,0]/2,train_imgs[img1,0]/2
R,G,B = R[0,0],G[0,0],B[0,0]
R+= diff*10

R,G,B = R.astype(np.uint8),G.astype(np.uint8),B.astype(np.uint8)
colored_img = np.swapaxes(np.swapaxes(np.array([R,G,B]),0,1),1,2)

cv2.imshow('hello',colored_img)

print(np.sum(np.abs(train_imgs[img1,0]-train_imgs[img2,0])))
print(np.sum(np.abs(train_imgs[img1,0]-train_imgs[img2,0]))/np.sum(train_imgs[img1,0]))