# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:33:33 2016

@author: sicarbonnell

defines model through segment() function
"""
import Utils.layers as layers_perso

from keras.models import Model
import keras.layers as layers
from keras.layers import merge, Dense, AveragePooling2D, MaxPooling2D, Activation, Flatten, Reshape, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU

def activation(other_choice = None):
    if other_choice == 'sigmoid':
        return Activation('sigmoid')
    else:
        return LeakyReLU(alpha = 0.2)

def convolutional(nbfilters,fsize1,fsize2,inp,pad = True,subsample = (1,1),batchnorm = True,other_activation = None):
    # fsize1 and fsize2 must be the same, and must be odd numbers
    if pad and fsize1 > 1: # double check
        inp = layers.ZeroPadding2D(padding=(int((fsize1-1)/2), int((fsize2-1)/2)))(inp)
    conv = layers.Convolution2D(nbfilters,fsize1,fsize2,border_mode = 'valid',subsample=subsample)(inp)
    if batchnorm:
        conv = layers.BatchNormalization(mode = 0,axis = 1)(conv)
    if other_activation is not None:
        conv = activation(other_activation)(conv)
    else:
        conv = activation()(conv)
    return conv


def local(nbfilters,fsize1,fsize2,inp,pad = True,subsample = (1,1), batchnorm =True, fast = True):
    if pad:
        inp = layers.ZeroPadding2D(padding=(int((fsize1-1)/2), int((fsize2-1)/2)))(inp)
    if not fast:
        lconv = layers.LocallyConnected2D(nbfilters,fsize1,fsize2,border_mode = 'valid',subsample=subsample)
    else:
        lconv = layers_perso.LocallyConnected2D_fast(nbfilters,fsize1,fsize2,border_mode = 'valid',subsample=subsample)
    conv = lconv((inp))
    if batchnorm:
        conv = layers_perso.BatchNormalization_local(lconv,conv)
    return activation()(conv)

def full(nb_neurons,inp,regularizer = None,drop = 0.5,sigmoid = False,name = None):
    full = Dense(nb_neurons)(inp)
    if regularizer != 'dropout':
        full = layers.BatchNormalization(mode = 0,axis = 1)(full)
    if sigmoid:
        full = activation('sigmoid')(full)
    else:
        full = activation()(full)
    if regularizer == 'dropout':
        full = layers.Dropout(drop)(full)
    return full
    

def segment(nb_rows = 160,nb_cols = 224):
    # a / b stands for global/local branches
    # first number is input size level, second is inner layer index
    inputs = layers.Input((1, nb_rows, nb_cols))
    up = []
    
    m1a = merge([convolutional(16,9,9,inputs),convolutional(8,7,7,inputs),convolutional(8,5,5,inputs)],\
    mode='concat', concat_axis=1)
    m1b = merge([convolutional(16,9,9,inputs),convolutional(8,7,7,inputs),convolutional(8,5,5,inputs)],\
    mode='concat', concat_axis=1)
    l1 = merge([MaxPooling2D(pool_size = (4,4))(m1a), AveragePooling2D(pool_size = (4,4))(m1b)],mode='concat', concat_axis=1)
    
    l211= convolutional(128,3,3,l1)
    l212= convolutional(128,3,3,l1)
    l21 = merge([l211,l212],mode='concat', concat_axis=1)
    l221 = convolutional(128,3,3,l21)
    l222 = convolutional(128,3,3,l21)
    l22 = merge([l221,l222],mode='concat', concat_axis=1)
    l22_pooled = MaxPooling2D(pool_size = (2,2))(l22)
    
    up.append(l212) 
    up.append(l222)
    
    l3a0 = convolutional(256,3,3,l22_pooled)
    l3a2 = convolutional(256,3,3,l3a0)
    l3a_pooled = MaxPooling2D(pool_size = (2,2))(l3a2)
    l3b0 = convolutional(256,3,3,l22_pooled) 
    l3b_pooled = MaxPooling2D(pool_size = (2,2))(l3b0)
    
    up.append(UpSampling2D((2,2))(l3a2))
    up.append(UpSampling2D((2,2))(l3b0))
    
    l4a1 = l3a_pooled
    l4a21 = convolutional(256,3,3,l4a1)
    l4a2 = merge([convolutional(256,1,1,l4a1,False),l4a21],mode='concat', concat_axis=1) 
    l4a2 = MaxPooling2D(pool_size = (2,2))(l4a2)
    l4b1 = merge([convolutional(64,1,1,l4a1,False),convolutional(256,3,3,l4a1)],mode='concat', concat_axis=1)   
    l4b1 = MaxPooling2D(pool_size = (2,2))(l4b1)
    l4b21 = local(32,1,1,l3b_pooled,False)
    l4b22 = local(32,2,2,l4b21,False, subsample = (2,2))
    l4b = merge([l4b1,l4b22],mode='concat', concat_axis=1)
    
    up.append(UpSampling2D((4,4))(l4a21))
    
    l51a = convolutional(256,3,3,l4a2)
    l52a1 = convolutional(128,1,1,l51a,False)
    l52a2 = convolutional(128,1,1,l51a,False)
    l52a = merge([MaxPooling2D(pool_size = (5,7))(l52a1),AveragePooling2D(pool_size = (5,7))(l52a2)],mode='concat', concat_axis=1)
    l51b = local(32,1,1,l4b, pad = False)
    l52b = local(32,3,3,l51b,fast = False)
    
    up.append(UpSampling2D((8,8))(l4b22))
    up.append(UpSampling2D((8,8))(l52a1))
    up.append(UpSampling2D((8,8))(l52b))
    
    l53 = merge([Flatten()(l52a),Flatten()(l52b)],mode='concat', concat_axis=1)
    l61 = full(128,l53)
    l62 = full(16,l61)
    
    up.append(UpSampling2D((40,56))(Reshape([16,1,1])(l62)))
    
    # unique value that should learn when a nerve is present on the image
    presence = Reshape([1,1,1])(full(1,l62,sigmoid = True))
    presence = UpSampling2D((40,56))(presence)
    
    # merge all and add layer for segmentation output
    f1 = merge(up,mode='concat', concat_axis=1)
    f2 = layers_perso.SemiShared(1,(8,8))(f1)
    f2 = layers_perso.BatchNormalization_local(f2._keras_history[0],f2)
    output = activation('sigmoid')(f2)
    
    # multiply predicted mask with predicted presence of a nerve
    output = merge([presence,output],mode ='mul')
    
    return Model(input=inputs, output = output)