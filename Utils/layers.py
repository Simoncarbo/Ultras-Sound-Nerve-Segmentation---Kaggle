# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:10:48 2016

@author: sicarbonnell

Code from: https://github.com/joelthchao/keras/blob/master/keras/layers/local.py
"""

# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import backend as K
from keras.layers import activations, initializations, regularizers, constraints, Reshape, BatchNormalization
from keras.layers.convolutional import conv_output_length
from keras.engine import Layer, InputSpec

def BatchNormalization_local(layer,tensor):
    shape = layer.output_shape
    return Reshape(shape[1:])(BatchNormalization(mode = 0,axis = 1)(Reshape((shape[1]*shape[2]*shape[3],))(tensor)))
    