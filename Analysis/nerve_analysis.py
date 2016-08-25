# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 16:32:05 2016

@author: sicarbonnell

Check nerve presence statistcs
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Utils.import_data import load_train_data


#import and pre-process data
train_imgs, train_masks, train_index = load_train_data()
del train_imgs
train_masks = train_masks/255 # convert to 0,1

# compute max size and put info in dataframe, levels ['subject','image'], column 'mask_size'
mask_size = np.sum(train_masks,(2,3),dtype = np.int32)
mask_sizes = pd.DataFrame(mask_size,columns = ['mask_size'],index = pd.MultiIndex.from_arrays([train_index[:,0],train_index[:,1]],names = ['subject','image']))
mask_sizes.sort_index(inplace = True)

#computes ratio of nerve presence for each subject
nerve_ratio = mask_sizes['mask_size'].groupby(level = 'subject').agg({'ratio': lambda x: np.sum(x>0)/len(x),'nb' :len})

#computes mean ratio (must be weighted by nb of images per user)
print('probability of nerve presence: '+str(np.sum(nerve_ratio['ratio']*nerve_ratio['nb'])/np.sum(nerve_ratio['nb']))) 

#plot nerve presence ratio histogram
plt.figure()
plt.hist(nerve_ratio['ratio'],bins = 50)

#computes mean mask
mean_mask = np.mean(train_masks[np.where(mask_size>0)],axis = 0)
plt.figure()
plt.hist(mean_mask.flatten(),bins = 50)