# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 16:53:57 2016

@author: sicarbonnell

data partitioning and ordering
data generation
"""
import Utils.processing as proc

import pandas as pd
import numpy as np
import numpy.random as random

def validation_split(train_index, nbsubjects):
    ordering = pd.DataFrame(data = np.arange(len(train_index[:,0])),columns = ['initial_order'],index = pd.MultiIndex.from_arrays([train_index[:,0],train_index[:,1]],names = ['subject','image']))
    ordering.sort_index(inplace = True)
    
    subjects = ordering.index.get_level_values('subject').unique()
    validation_subjects = random.choice(subjects,size = nbsubjects,replace = False)
    
    val_indexes = ordering.loc[(validation_subjects,slice(None)),'initial_order']
    train_indexes = np.setdiff1d(np.arange(len(train_index[:,0])),val_indexes)
    
    return train_indexes,val_indexes
    
def train_ordering(train_imgs, train_masks, train_index):
    ordering = pd.DataFrame(data = np.arange(len(train_index[:,0])),columns = ['initial_order'],index = pd.MultiIndex.from_arrays([train_index[:,0],train_index[:,1]],names = ['subject','image']))
    ordering.sort_index(inplace = True, level = 'subject')
    
    #shuffle intra subject
    ordering = ordering.groupby(level = 'subject').apply(lambda x:x.iloc[random.permutation(len(x))])
    ordering.index = ordering.index.droplevel(0)
    
    #create new column with new subject image order
    ordering['new'] = 0
    ordering['new'] = ordering['new'].groupby(level = 'subject').transform(lambda x:np.arange(len(x)).T)
    
    #take all first images per subject and so on
    final_ordering = np.array([])
    for i in ordering['new'].unique():
        idx = ordering.loc[ordering['new'] == i,'initial_order'] # indexes of i'th image for each user after shuffling
        idx = idx.iloc[random.permutation(len(idx))] # shuffle users in batch
        final_ordering = np.hstack((final_ordering,idx.values))
    final_ordering = final_ordering.astype(int)
    train_imgs , train_masks, train_index = train_imgs[final_ordering], train_masks[final_ordering], train_index[final_ordering]
    return train_imgs , train_masks, train_index

def data_generator_segment(rows,cols,nb_rows_mask, nb_columns_mask):
    augmentation_factor = 1 # number of generated images from one original training image
    
    def data_generator(train_imgs, train_masks,train_index, batch_size): # batch size without augmentation!
        while 1:
            ##shuffle and order data
            train_imgs, train_masks, train_index = train_ordering(train_imgs, train_masks, train_index)    
                        
            count = 0
            batch_x = np.ndarray((batch_size*augmentation_factor,1,rows,cols))
            batch_y = np.ndarray((batch_size*augmentation_factor,1,nb_rows_mask, nb_columns_mask))
            for i in range(len(train_imgs)):
                # augment and add to batch
                v = 0 # counts number of images added per image
                batch_x[count*augmentation_factor+v] = train_imgs[i]
                batch_y[count*augmentation_factor+v] = train_masks[i]
                
#                batch_x[count*augmentation_factor+v,0] = proc.transform(train_imgs[i,0])
                ## TO MODIFY SUCH THAT MASKS ARE MODIFED TOO
                v+=1
                
                count += 1
                if count >= batch_size:                    
                    yield (batch_x, batch_y)
                    count = 0
                    batch_x = np.ndarray((batch_size*augmentation_factor,1,rows,cols))
                    batch_y = np.ndarray((batch_size*augmentation_factor,1,nb_rows_mask, nb_columns_mask))
                elif i == len(train_imgs)-1:
                    batch_x = np.delete(batch_x,np.arange(count*augmentation_factor,batch_size*augmentation_factor),axis = 0) 
                    batch_y = np.delete(batch_y,np.arange(count*augmentation_factor,batch_size*augmentation_factor),axis = 0)
                    yield (batch_x, batch_y)
    return augmentation_factor, data_generator