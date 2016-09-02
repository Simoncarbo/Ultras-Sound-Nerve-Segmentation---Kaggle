# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 12:01:26 2016

@author: sicarbonnell

predicts masks on test set, creates submission file and saves predicted masks in Predictions folder
"""
import numpy as np
np.random.seed(172)
import cv2

from keras.models import model_from_json

import Utils.processing as proc
from Utils.import_data import load_test_data
from Utils.layers import SemiShared, LocallyConnected2D_fast
np.random.seed(172)

model_name = 'Saved/model'
submission_name = 'submission.csv'

epochs = [14,17,19,22,24,29,44,46,48]
#epochs = [21,22,23,29,30,31,32,34,35,49] #for model1

nb_rows_small, nb_cols_small = 160,224

def predict_and_save():
    imgs, index = load_test_data()
    nb_rows, nb_cols = imgs.shape[-2:]
    
    argsort = np.argsort(index)
    index = index[argsort]
    imgs = imgs[argsort]
    
    m = np.load(model_name+'_m.npy')
    st = np.load(model_name+'_st.npy')
    
    # preprocessing
    imgs = proc.preprocessing_test(imgs, m,st,(nb_rows_small, nb_cols_small))
    model = model_from_json(open(model_name+'.json').read(),{'SemiShared':SemiShared,'local_fast':LocallyConnected2D_fast})
    
    for epoch in epochs:
        weights_name = model_name+'_weights_epoch_'+str(epoch)+'.hdf5'
        model.load_weights(weights_name)
            
        print('predicting masks...')
        masks = model.predict(imgs,verbose = 1)
        
        masks = proc.postprocess_masks(masks,(nb_rows,nb_cols))
        
        np.save(model_name+'_pred_epoch'+str(epoch)+'.npy', masks)
    return masks.shape, index
        
def combine(shape):
    sum_masks = np.ndarray(shape, dtype = np.float32)
    for epoch in epochs:
        masks = np.load(model_name+'_pred_epoch'+str(epoch)+'.npy')
        print('model at epoch '+str(epoch)+' predicts '+str(np.mean(np.max(np.max(masks,axis =-1),axis = -1)))+' nerves')
        sum_masks += masks
        
    return sum_masks

def create_submission_file(masks, shape, index):
    def run_length_enc(x):
        from itertools import chain
        x = x.transpose().flatten()
        y = np.where(x > 0)[0]
        if len(y) < 10:  # consider as empty
            return ''
        z = np.where(np.diff(y) > 1)[0]
        start = np.insert(y[z+1], 0, y[0])
        end = np.append(y[z], y[-1])
        length = end - start
        res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
        res = list(chain.from_iterable(res))
        return ' '.join([str(r) for r in res])
        
    total = shape[0]
    rles = []
    for i in range(total):
        img = masks[i, 0]
        rle = run_length_enc(img)
        rles.append(rle)
        
    first_row = 'img,pixels'
    with open(submission_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(index[i]) + ',' + rles[i]
            f.write(s + '\n')

def write_masks(masks, index):
    for i,img in enumerate(masks):
        img = cv2.threshold((img[0]*255).astype(np.uint8), 127.5, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite('Predictions/'+str(index[i])+'_mask.png',img)
    
shape, index = predict_and_save()
sum_masks = combine(shape)
masks = np.round(((sum_masks-2)/len(epochs))).astype(np.uint8)
print('combined model predicts '+str(np.mean(np.max(np.max(masks,axis =-1),axis = -1)))+' nerves')
create_submission_file(masks, shape, index)
write_masks(masks, index)



        


