# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 15:34:51 2016

@author: sicarbonnell

Image processing methods (data augmentation and pre-processing)
"""
import numpy as np
import numpy.random as random
import skimage.transform as tf
import cv2

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def preprocessing_imgs(train_imgs, reduced_size = None):
    # resizing    
    if reduced_size is not None:
        train_imgs_p = np.ndarray((train_imgs.shape[0], train_imgs.shape[1]) + reduced_size, dtype=np.float32)
        for i in range(train_imgs.shape[0]):
            train_imgs_p[i, 0] = cv2.resize(train_imgs[i, 0], (reduced_size[1], reduced_size[0]), interpolation=cv2.INTER_CUBIC) # INVERSE ORDER! cols,rows
    else:
        train_imgs_p = train_imgs.astype(np.float32)
        
    # ZMUV normalization
    m = np.mean(train_imgs_p).astype(np.float32)
    train_imgs_p -= m
    st = np.std(train_imgs_p).astype(np.float32)
    train_imgs_p /= st
    
    return train_imgs_p,m,st

def preprocessing_masks(train_masks, reduced_size = None):       
    #resizing
    if reduced_size is not None:
        train_masks_p = np.ndarray((train_masks.shape[0], train_masks.shape[1]) + reduced_size, dtype=np.uint8)
        for i in range(train_masks.shape[0]):
            train_masks_p[i, 0] = cv2.resize(train_masks[i, 0], (reduced_size[1], reduced_size[0]), interpolation=cv2.INTER_CUBIC)
    else:
        train_masks_p = train_masks
    # to [0,1]
    train_masks_p = (train_masks_p / 255).astype(np.uint8)
    return train_masks_p
    
def preprocessing_test(test_imgs, m,st,reduced_size = None):
    # resizing    
    if reduced_size is not None:
        test_imgs_p = np.ndarray((test_imgs.shape[0], test_imgs.shape[1]) + reduced_size, dtype=np.float32)
        for i in range(test_imgs.shape[0]):
            test_imgs_p[i, 0] = cv2.resize(test_imgs[i, 0], (reduced_size[1], reduced_size[0]), interpolation=cv2.INTER_CUBIC) # INVERSE ORDER! cols,rows
    else:
        test_imgs_p = test_imgs.astype(np.float32)
        
    # ZMUV normalization
    test_imgs_p -= m
    test_imgs_p /= st
    
    return test_imgs_p

def postprocess_masks(masks,new_size = None):
    if new_size is not None:
        masks_p = np.ndarray((masks.shape[0], masks.shape[1]) + new_size, dtype=np.float32)
        for i in range(masks.shape[0]):
            masks_p[i, 0] = cv2.resize(masks[i, 0], (new_size[1],new_size[0]), interpolation=cv2.INTER_LINEAR)
    else:
        masks_p = masks.copy()
    
    masks_p[np.where(np.sum(np.sum(masks_p,axis = -1),axis = -1)[:,0]<4000)] = 0
    
    for i in range(masks.shape[0]):
        masks_p[i,0] = cv2.blur(masks_p[i,0],(30,30))
        
    masks_p = np.round(masks_p)
    
    for i in range(masks.shape[0]):
        blurred = cv2.blur(masks_p[i,0],(100,100))
        masks_p[(i,0)+np.where(blurred<0.1)] =0
    
    masks_p[np.where(np.sum(np.sum(masks_p,axis = -1),axis = -1)[:,0]<1500)] = 0
    
    return masks_p.astype(np.uint8)
    
def transform(image): #translate, shear, stretch, flips?
    rows,cols = image.shape
    
    angle = random.uniform(-1.5,1.5)
    center = (rows / 2 - 0.5+random.uniform(-50,50), cols / 2 - 0.5+random.uniform(-50,50))
    def_image = tf.rotate(image, angle = angle, center = center,clip = True, preserve_range = True,order = 5)
    
    alpha = random.uniform(0,5)
    sigma = random.exponential(scale = 5)+2+alpha**2
    def_image = elastic_transform(def_image, alpha, sigma)
    
    def_image = def_image[10:-10,10:-10]
    
    return def_image
    
# sigma: variance of filter, fixes homogeneity of transformation 
#    (close to zero : random, big: translation)
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       
       Code taken from https://gist.github.com/fmder/e28813c1e8721830ff9c
       slightly modified
    """
    min_im = np.min(image)
    max_im = np.max(image)
    
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0)
    dx = dx/np.max(dx)* alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0)
    dy = dy/np.max(dy)* alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    image_tfd = map_coordinates(image,indices,order=3).reshape(shape)
    image_tfd[image_tfd>max_im] = max_im
    image_tfd[image_tfd<min_im] = min_im
    
    return image_tfd
    