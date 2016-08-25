# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:24:14 2016

@author: sicarbonnell
"""
import numpy as np
np.random.seed(172)

from keras.models import model_from_json

from Utils.import_data import load_train_data
import Utils.processing as proc
import Utils.data_management as dm
from Utils.custom_callbacks import LossHistory
import models

smooth = 1e-5#e-1
def dice_coef_np(y_true, y_pred):
    intersection = np.reshape(np.sum(np.sum(y_true * y_pred,axis = -1),axis = -1),[y_true.shape[0]])
    sum_pred = np.reshape(np.sum(np.sum(y_pred,axis = -1),axis = -1),[y_true.shape[0]])    
    sum_true = np.reshape(np.sum(np.sum(y_true,axis = -1),axis = -1),[y_true.shape[0]])

    return -np.mean((2. * intersection  + smooth) / (sum_pred + sum_true + smooth))
    
train_imgs, train_masks, train_index = load_train_data()
train_masks= (train_masks/255).astype(np.uint8)
nb_rows,nb_cols = train_imgs.shape[-2], train_imgs.shape[-1]

train_imgs,m1,st1 = proc.preprocessing_imgs(train_imgs,(160, 224)) 
model1 = model_from_json(open('Saved/segment_coarse.json').read()) # il faut completement changer le processing
model1.load_weights('Saved/segment_coarse.h5')
train_masks_coarse = model1.predict(train_imgs,verbose = 1,batch_size = 42)
del train_imgs

train_masks_coarse,m_masks,st_masks = proc.preprocessing_imgs(train_masks_coarse,(nb_rows,nb_cols)) 

print('training set error: '+str(dice_coef_np(train_masks,train_masks_coarse*st_masks+m_masks)))
tres = (train_masks_coarse*st_masks+m_masks).copy()
tres[tres>= 0.5] = 1
tres[tres< 0.5] = 0
print('training set error with treshold: '+str(dice_coef_np(train_masks,tres)))

subjects_val = [ 7, 18, 31, 33, 44]
val_i = np.in1d(train_index[:,0],subjects_val)
valid_masks_coarse, valid_masks = train_masks_coarse[val_i], train_masks[val_i]
print('training set error: '+str(dice_coef_np(valid_masks,valid_masks_coarse*st_masks+m_masks)))
tres = (valid_masks_coarse*st_masks+m_masks).copy()
tres[tres>= 0.5] = 1
tres[tres< 0.5] = 0
print('training set error with treshold: '+str(dice_coef_np(valid_masks,tres)))