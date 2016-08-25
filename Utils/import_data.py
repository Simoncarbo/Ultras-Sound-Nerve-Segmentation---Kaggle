# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:35:48 2016

@author: sicarbonnell

From Marko Jocic
"""

import os
import numpy as np

import cv2

data_path = 'Data/'

image_rows = 420
image_cols = 580


def create_train_data():
    train_data_path = data_path+'train/'
    images = os.listdir(train_data_path) #list of all files
    total = int(len(images) / 2) # number of training examples

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total,2 ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        name = image_name.split('.')[0]  #image name without .tif
        ids = name.split('_') # subject and image index
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = cv2.imread(train_data_path+ image_name, cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(train_data_path+ image_mask_name, cv2.IMREAD_GRAYSCALE)

        img = np.array([img])
        img_mask = np.array([img_mask])
        
        imgs[i] = img
        imgs_mask[i] = img_mask
        imgs_id[i] = ids

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(data_path+'train_imgs.npy', imgs)
    np.save(data_path+'train_masks.npy', imgs_mask)
    np.save(data_path+'train_index.npy', imgs_id)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load(data_path+'train_imgs.npy')
    imgs_mask_train = np.load(data_path+'train_masks.npy')
    imgs_train_id = np.load(data_path+'train_index.npy')
    return imgs_train, imgs_mask_train, imgs_train_id


def create_test_data():
    test_data_path = data_path+'test/'
    images = os.listdir(test_data_path)
    total = len(images)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = cv2.imread(test_data_path+ image_name, cv2.IMREAD_GRAYSCALE)
        
        img = np.array([img])
        
        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(data_path+'test_imgs.npy', imgs)
    np.save(data_path+'test_index.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load(data_path+'test_imgs.npy')
    imgs_test_id = np.load(data_path+'test_index.npy')
    return imgs_test, imgs_test_id

if __name__ == '__main__':
    create_train_data()
    create_test_data()