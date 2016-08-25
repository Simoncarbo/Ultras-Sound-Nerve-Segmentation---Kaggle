# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 11:23:19 2016

@author: sicarbonnell
"""
import numpy as np
#np.random.seed(172)
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from keras.models import model_from_json

import Utils.data_management as dm

#
model = model_from_json(open('Saved/segment_model.json').read()) # il faut completement changer le processing
#model.load_weights('Saved/segment_coarse.h5')
model.load_weights('Saved/weights_epoch_26.hdf5')
#
#model.compile(optimizer =Adam(lr=1e-2), loss = dice_coef,metrics = [dice_coef_tresh,dice_coef_with,pres_acc,pres_acc_with, pres_acc_without])
#print(model.evaluate(valid_imgs,valid_masks))

#model2 = model_from_json(open('Saved/segment_coarse2.json').read()) # il faut completement changer le processing
#model2.load_weights('Saved/segment_coarse2.h5')

smooth = 1
def dice_coef_np(y_true, y_pred):
    intersection = np.reshape(np.sum(np.sum(y_true * y_pred,axis = -1),axis = -1),[y_true.shape[0]])
    sum_pred = np.reshape(np.sum(np.sum(y_pred,axis = -1),axis = -1),[y_true.shape[0]])    
    sum_true = np.reshape(np.sum(np.sum(y_true,axis = -1),axis = -1),[y_true.shape[0]])

    return -np.mean((2. * intersection  + smooth) / (sum_pred + sum_true + smooth))
    
def dice_coef_with_np(y_true, y_pred):
    intersection = np.reshape(np.sum(np.sum(y_true * y_pred,axis = -1),axis = -1),[y_true.shape[0]])
    sum_pred = np.reshape(np.sum(np.sum(y_pred,axis = -1),axis = -1),[y_true.shape[0]])    
    sum_true = np.reshape(np.sum(np.sum(y_true,axis = -1),axis = -1),[y_true.shape[0]])
    
    presence_true = np.max(np.max(y_true,axis = -1),axis = -1)
    presence_pred = np.max(np.max(y_pred,axis = -1),axis = -1)
    with_nerve = (np.equal(np.round(presence_pred),1) * np.equal(presence_true,1)).nonzero()[0]
    return -np.mean((2. * intersection[with_nerve]) / (sum_pred[with_nerve] + sum_true[with_nerve]))

#x = train_imgs_small[train_i]
#y = train_nerve_presence
#y2 = train_masks_small[train_i]
#index = train_index[train_i]

x = train_imgs[val_i]
y = valid_nerve_presence
y2 = train_masks[val_i]
index = train_index[val_i]

predictions = model.predict(x,verbose = 1,batch_size = 32)
#predictions2 = model2.predict(x,verbose = 1,batch_size = 32)
#predictions = (predictions1+predictions2)/2
#predictions = np.maximum(predictions1,predictions2)
#predictions = np.minimum(predictions1,predictions2)
#out2 = np.reshape(predictions[1],[predictions[1].shape[0],1,1,1])

presence_pred = np.reshape(np.max(np.max(predictions,axis = -1),axis = -1),[predictions.shape[0],])
tres = predictions.copy()
tres[tres>= 0.5] = 1
tres[tres< 0.5] = 0

with_nerve = np.where(y==1)
without_nerve = np.where(y==0)
print(np.mean(np.equal(np.round(presence_pred),y)))
#print(np.mean(np.equal(np.round(presence_pred[with_nerve]),y[with_nerve])))
#print(np.mean(np.equal(np.round(presence_pred[without_nerve]),y[without_nerve])))
print(dice_coef_np(y2,tres))
#print(dice_coef_with_np(y2,tres))

#plt.hist(np.reshape(predictions[0,0],[40*56]),bins = 100)

#for i,img in enumerate(predictions):
#    img = cv2.threshold((img[0]*255).astype(np.uint8), 0.5, 255, cv2.THRESH_BINARY)[1]
#    img = cv2.resize(img, (580,420), interpolation=cv2.INTER_CUBIC)
#    cv2.imwrite('predictions/valid/'+str(index[i,0])+'_'+str(index[i,1])+'_valid.png',img)
