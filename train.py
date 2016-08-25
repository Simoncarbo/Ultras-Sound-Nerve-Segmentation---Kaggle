# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:58:40 2016

@author: sicarbonnell

training and accuracy tracking
"""
import numpy as np
np.random.seed(172)

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

import Utils.processing as proc
import Utils.data_management as dm
from Utils.import_data import load_train_data
import models
np.random.seed(172)

nb_rows_small, nb_cols_small = 160,224
nb_rows_mask_small, nb_cols_mask_small = 40,56

nb_subjects_val = 10
save_models = True

nerve_presence_wanted = 0.45

smooth = 1
def dice_tresh(y_true, y_pred):
    y_pred = K.round(y_pred)
    
    intersection = K.sum(K.sum(y_true * y_pred,axis = -1),axis = -1)
    sum_pred = K.sum(K.sum(y_pred,axis = -1),axis = -1)
    sum_true = K.sum(K.sum(y_true,axis = -1),axis = -1)
    
    return -K.mean((2. * intersection  + smooth) / (sum_true + sum_pred + smooth))
        
def pres_acc(y_true, y_pred):
    true = (K.max(K.max(y_true,axis = -1),axis = -1))
    pred = (K.max(K.max(y_pred,axis = -1),axis = -1))
    return K.mean(K.equal(K.round(pred),K.round(true)))
        
def train_segment(train_imgs, train_masks, train_index,train_i,val_i,factor,factor_val):
    def dice_coef(y_true, y_pred):
        intersection = K.sum(K.sum(y_true * y_pred,axis = -1),axis = -1)
        sum_pred = K.sum(K.sum(y_pred,axis = -1),axis = -1)
        sum_true = K.sum(K.sum(y_true,axis = -1),axis = -1)
        
        weighting = K.greater_equal(sum_true,1)*factor+1
        return -K.mean(weighting*(2. * intersection  + smooth) / (sum_true + sum_pred + smooth))
    def dice_coef_wval(y_true, y_pred):
        intersection = K.sum(K.sum(y_true * y_pred,axis = -1),axis = -1)
        sum_pred = K.sum(K.sum(y_pred,axis = -1),axis = -1)
        sum_true = K.sum(K.sum(y_true,axis = -1),axis = -1)
        
        weighting = K.greater_equal(sum_true,1)*factor_val+1
        return -K.mean(weighting*(2. * intersection  + smooth) / (sum_true + sum_pred + smooth))
      
    model = models.segment()
    
    model.compile(optimizer =Adam(lr=1e-2), loss = dice_coef,metrics = [dice_coef_wval,dice_tresh,pres_acc])
    
    augmentation_ratio, data_generator = dm.data_generator_segment(nb_rows_small, nb_cols_small,nb_rows_mask_small, nb_cols_mask_small)
    
    def schedule(epoch):
        if epoch<=5:
            return 1e-2
        elif epoch<=10:
            return 5e-3
        elif epoch<=25:
            return 2e-3
        elif epoch<=40:
            return 1e-3
        else:
            return 5e-4
    lr_schedule= LearningRateScheduler(schedule)
    modelCheck = ModelCheckpoint('Saved/model2_weights_epoch_{epoch:02d}.hdf5', verbose=0, save_best_only=False)
    
    print('training starts...')
    epoch_history = model.fit_generator(\
    data_generator(train_imgs[train_i], train_masks[train_i], train_index[train_i],batch_size = len(np.unique(train_index[train_i,0]))), \
    samples_per_epoch = augmentation_ratio*len(train_i),nb_epoch = 50, callbacks = [lr_schedule,modelCheck], \
    validation_data = (train_imgs[val_i],train_masks[val_i]),max_q_size=10)
        
    return model, epoch_history
    
#==============================================================================
# Data importation and processing
#==============================================================================
train_imgs, train_masks, train_index = load_train_data()
nb_rows, nb_cols = train_imgs.shape[-2:]
nb_subjects = len(np.unique(train_index[:,0]))

train_imgs,m,st = proc.preprocessing_imgs(train_imgs,(nb_rows_small, nb_cols_small))
train_masks = proc.preprocessing_masks(train_masks, (nb_rows_mask_small, nb_cols_mask_small))

train_i, val_i = dm.validation_split(train_index,nb_subjects_val)
validation_subjects = np.unique(train_index[val_i,0])
print('Validation set contains '+str(np.around(len(val_i)/len(train_imgs),2))+'% of initial training set')

train_nerve_presence = np.array([train_masks[i,0,:,:].max() for i in train_i])
valid_nerve_presence = np.array([train_masks[i,0,:,:].max() for i in val_i])

print('proportion of images with nerve in training set: '+str(np.around(np.mean(train_nerve_presence),4)))
print('proportion of images with nerve in validation set: '+str(np.around(np.mean(valid_nerve_presence),4)))
print('Data imported and processed...')

factor = nerve_presence_wanted/np.mean(train_nerve_presence)
print('loss for training examples with nerve will be multiplied by '+str(np.around(factor,3))+' to get '+str(nerve_presence_wanted)+' proportion')
factor_val = nerve_presence_wanted/np.mean(valid_nerve_presence)
print('loss for validation examples with nerve will be multiplied by '+str(np.around(factor_val,3))+' to get '+str(nerve_presence_wanted)+' proportion')
  
model,epoch_history = train_segment(train_imgs,train_masks,train_index,train_i,val_i,factor,factor_val)

if save_models:
    np.save('Saved/model_m.npy', m)
    np.save('Saved/model_st.npy', st)
    json_string = model.to_json()
    open('Saved/model.json', 'w').write(json_string)