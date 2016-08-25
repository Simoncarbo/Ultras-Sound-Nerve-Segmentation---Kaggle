# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:52:33 2016

@author: sicarbonnell
"""
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from Utils.import_data import load_train_data, load_test_data
import Utils.processing as proc

#import data
train_imgs, train_masks, train_index = load_train_data()
nb_rows, nb_columns = 72,96 # resize necessary for memory covariance matrix
train_imgs, train_masks = proc.preprocessing_train(train_imgs, train_masks, (nb_rows, nb_columns))

test_imgs, test_index = load_test_data()



del train_masks#, test_imgs

train_imgs = train_imgs.reshape([train_imgs.shape[0],train_imgs.shape[2]*train_imgs.shape[3]]).T

pca = PCA(n_components = 2)
pca.fit(train_imgs)

train_imgs_comp = pca.components_

plt.scatter(train_imgs_comp[0,:],train_imgs_comp[1,:])


train_means = np.mean(train_imgs,(2,3))
plt.scatter(train_means,np.zeros_like(train_means))

