# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 14:15:40 2016

@author: sicarbonnell

Custom callbacks
"""
import keras.callbacks

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []      
        
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('acc'))